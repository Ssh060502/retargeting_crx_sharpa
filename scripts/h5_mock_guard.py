"""Fail-closed discovery checks for the dedicated software-mock ROS domain."""
import os
import threading
import time


def require_mock_domain():
    if os.environ.get("ROS_DOMAIN_ID") != "187":
        raise RuntimeError("Mock replay requires ROS_DOMAIN_ID=187 in BOTH launch and replay terminals")
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
        raise RuntimeError("Mock replay requires ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST")


def validate_plugins(components):
    if not components or any(c.plugin_name != "mock_components/GenericSystem" for c in components):
        raise RuntimeError("Controller manager is not exclusively mock_components/GenericSystem")
    if any(c.state.label.lower() != "active" for c in components):
        raise RuntimeError("Mock hardware components must be active")


class MockGuard:
    """Read controller-manager state and hand parameters; never publish commands."""
    def __init__(self):
        require_mock_domain()
        import rclpy
        from rclpy.context import Context
        from controller_manager_msgs.srv import ListHardwareComponents
        from rcl_interfaces.srv import GetParameters
        self.rclpy = rclpy
        self.context = Context(); rclpy.init(args=[], context=self.context)
        self.node = rclpy.create_node("h5_mock_guard", context=self.context)
        self.arms = [self.node.create_client(ListHardwareComponents, f"/crx5ia/{s}/controller_manager/list_hardware_components") for s in ("left", "right")]
        self.hands = [self.node.create_client(GetParameters, f"/sharpa/{s}_hand/hand_node/get_parameters") for s in ("left", "right")]
        self.hardware_request, self.parameter_request = ListHardwareComponents, GetParameters
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.failure, self.last_ok = None, 0.
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        if not self.ready.wait(12.):
            self.close(); raise RuntimeError("Mock verification timed out")
        try:
            self.check()
        except BaseException:
            self.close(); raise

    def _call(self, client, request):
        if not client.wait_for_service(timeout_sec=2.):
            raise RuntimeError("Mock verification service missing: " + client.srv_name)
        future = client.call_async(request)
        deadline = time.monotonic() + 2.
        while not future.done() and time.monotonic() < deadline and not self.stop_event.is_set():
            self.rclpy.spin_once(self.node, timeout_sec=.05)
        if not future.done() or future.exception() is not None:
            raise RuntimeError("Mock verification service failed: " + client.srv_name)
        return future.result()

    def _verify(self):
        # Refuse ambiguous graph identities in the isolated domain.
        names = self.node.get_node_names_and_namespaces()
        for side in ("left", "right"):
            for name, ns in (("controller_manager", f"/crx5ia/{side}"), ("hand_node", f"/sharpa/{side}_hand")):
                if names.count((name, ns)) != 1:
                    raise RuntimeError("Missing or duplicate mock node: " + ns + "/" + name)
        for topic in ("/crx5ia/joint_targets", "/sharpa/left_hand/joint_command", "/sharpa/right_hand/joint_command"):
            publishers = self.node.get_publishers_info_by_topic(topic)
            if len(publishers) > 1 or any(p.node_name != "retargeting_sharpa_joint" for p in publishers):
                raise RuntimeError("Unexpected competing command publisher: " + topic)
        for client in self.arms:
            response = self._call(client, self.hardware_request.Request())
            validate_plugins(response.component)
        for client in self.hands:
            request = self.parameter_request.Request(); request.names = ["backend", "read_only"]
            response = self._call(client, request)
            values = response.values
            if len(values) != 2 or values[0].type != 4 or values[0].string_value != "mock" or values[1].type != 1 or values[1].bool_value:
                raise RuntimeError("Both hands must report backend=mock, read_only=false")

    def _loop(self):
        # Discovery is allowed to settle initially; failures latch once ready.
        startup = time.monotonic() + 10.
        while not self.stop_event.is_set():
            try:
                self._verify()
                self.last_ok = time.monotonic(); self.ready.set()
            except Exception as exc:
                if not self.ready.is_set() and time.monotonic() < startup:
                    self.stop_event.wait(.2); continue
                self.failure = str(exc); self.ready.set(); return
            self.stop_event.wait(.5)

    def check(self):
        if self.failure or time.monotonic() - self.last_ok > 3.:
            raise RuntimeError("Mock verification lost: " + str(self.failure or "stale verification"))

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10.)
        if self.thread.is_alive():
            raise RuntimeError("Mock guard worker did not stop")
        self.node.destroy_node(); self.context.try_shutdown()
