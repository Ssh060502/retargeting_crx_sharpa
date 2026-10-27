"""Single-arm EMPTY-HAND small-motion test, default offline preparation only.

This does not reproduce hammer contact or certify collision clearance. Other arm
and all finger joints receive NO commands from this script. Existing mock replay
and offline retarget entry points are unchanged.
"""
import argparse
import csv
import os
from pathlib import Path
import queue
import sys
import threading
import time
import numpy as np
from h5_arm_smoke import build_relative_test, anchor_path
from h5_replay_geometry import UrdfModel
from run_h5_replay import load_crx_side_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory",required=True)
    parser.add_argument("--side",required=True,choices=["left","right"])
    parser.add_argument("--mode",choices=["prepare","mock","real"],default="prepare")
    parser.add_argument("--frames",type=int,default=30)
    parser.add_argument("--out",default="outputs/hammer/arm_smoke_relative.npz")
    args=parser.parse_args()
    with np.load(args.trajectory,allow_pickle=False) as f:
        if "preview_only" in f and bool(f["preview_only"]):
            raise ValueError("Display-only entry preview cannot be used for motion")
        if args.side not in f["sides"].tolist():
            raise ValueError("Side absent from source trajectory")
        relative,rate=build_relative_test(f["qpos56"],args.side,args.frames)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,relative_joints=relative,rate=rate,side=args.side)
    print("Relative test:",args.side,"duration s",(len(relative)-1)/rate,"max excursion deg",np.rad2deg(np.abs(relative).max()))
    print("Only selected arm J1..J6. Empty hand, clear elevated starting pose required. No collision certification.")
    if args.mode=="prepare":
        print("Prepared only; no ROS imports or commands. File:",out);return
    expected_domain="187" if args.mode=="mock" else "87"
    if os.environ.get("ROS_DOMAIN_ID")!=expected_domain:
        parser.error("This mode requires explicit ROS_DOMAIN_ID="+expected_domain)
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE")!="LOCALHOST":
        parser.error("Require ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST")
    if not sys.stdin.isatty():
        parser.error("Motion requires an interactive operator terminal")
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from controller_manager_msgs.srv import ListHardwareComponents, ListControllers
    from rcl_interfaces.srv import GetParameters
    from rclpy.qos import qos_profile_sensor_data
    cfg=load_crx_side_config(args.side);model=UrdfModel(cfg["urdf_path"])
    bounds=model.joint_limits[[model.joint_index("J"+str(i)) for i in range(1,7)]]
    context=Context();rclpy.init(args=[],context=context)
    node=rclpy.create_node("h5_arm_smoke",context=context)
    executor=SingleThreadedExecutor(context=context);executor.add_node(node)
    topic="/crx5ia/joint_targets";expected=[args.side+"_J"+str(i) for i in range(1,7)]
    feedback={"q":None,"received":0.,"stamp":0,"advanced":0.}
    publisher=None;armed=False;reason="not started"
    def receive(msg):
        if len(msg.name)!=len(msg.position) or len(set(msg.name))!=len(msg.name):return
        by_name=dict(zip(msg.name,msg.position))
        if not all(n in by_name for n in expected):return
        q=np.array([by_name[n] for n in expected]);stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        age=(node.get_clock().now().nanoseconds-stamp)/1e9
        if not np.isfinite(q).all() or stamp<=0 or not -.02<=age<=.3 or stamp<feedback["stamp"]:return
        now=time.monotonic()
        if stamp>feedback["stamp"]:feedback["advanced"]=now
        feedback.update(q=q,received=now,stamp=stamp)
    node.create_subscription(JointState,f"/crx5ia/{args.side}/joint_states",receive,qos_profile_sensor_data)
    hardware=node.create_client(ListHardwareComponents,f"/crx5ia/{args.side}/controller_manager/list_hardware_components")
    controllers=node.create_client(ListControllers,f"/crx5ia/{args.side}/controller_manager/list_controllers")
    parameters=node.create_client(GetParameters,"/crx5ia/joint_interpolation/get_parameters")
    def call(client,request,timeout=2.):
        if not client.wait_for_service(timeout_sec=timeout):raise RuntimeError("Missing service: "+client.srv_name)
        future=client.call_async(request);until=time.monotonic()+timeout
        while not future.done() and time.monotonic()<until:executor.spin_once(timeout_sec=.01)
        if not future.done() or future.exception():raise RuntimeError("Service failed: "+client.srv_name)
        return future.result()
    def verify_stack():
        names=node.get_node_names_and_namespaces()
        for name,ns in [("controller_manager",f"/crx5ia/{args.side}"),("joint_interpolation","/crx5ia")]:
            if names.count((name,ns))!=1:raise RuntimeError("Missing/ambiguous node: "+ns+"/"+name)
        components=call(hardware,ListHardwareComponents.Request()).component
        plugin="mock_components/GenericSystem" if args.mode=="mock" else "fanuc_robot_driver/FanucHardwareInterface"
        if not components or any(c.plugin_name!=plugin or c.state.label.lower()!="active" for c in components):
            raise RuntimeError("Hardware plugin/state does not match requested mode")
        listing=call(controllers,ListControllers.Request()).controller
        if not any(c.name=="forward_position_controller" and c.state=="active" for c in listing):
            raise RuntimeError("Selected forward_position_controller not active")
        req=GetParameters.Request();req.names=["method","input_rate_hz"]
        values=call(parameters,req).values
        if len(values)!=2 or values[0].string_value!="linear" or abs(values[1].double_value-rate)>1e-6:
            raise RuntimeError("Launch requires method:=linear input_rate_hz:=50.0")
    def actual():
        now=time.monotonic()
        age=(node.get_clock().now().nanoseconds-feedback["stamp"])/1e9
        if feedback["q"] is None or now-feedback["received"]>.3 or now-feedback["advanced"]>.3 or not -.02<=age<=.3:
            raise RuntimeError("Selected-arm feedback stale, missing or frozen")
        return feedback["q"].copy()
    def verify_routes():
        pubs=node.get_publishers_info_by_topic(topic)
        if any(p.node_name!="h5_arm_smoke" for p in pubs) or len(pubs)>(1 if publisher else 0):
            raise RuntimeError("Competing command publisher; stop keyboard/teleop/replay senders")
        subs=node.get_subscriptions_info_by_topic(topic)
        if len(subs)!=1 or subs[0].node_name!="joint_interpolation":
            raise RuntimeError("Missing/ambiguous interpolation subscriber")
    def send(q):
        msg=JointState();msg.header.stamp=node.get_clock().now().to_msg();msg.name=expected;msg.position=q.tolist();publisher.publish(msg)
    controls=queue.Queue()
    def keyboard():
        for line in sys.stdin:
            if line.strip().lower()=="q":controls.put("q");return
        controls.put("q")
    try:
        until=time.monotonic()+5.
        while time.monotonic()<until:
            executor.spin_once(timeout_sec=.05)
            if feedback["q"] is not None:break
        verify_stack();verify_routes();start=actual()
        path=anchor_path(relative,start,bounds[:,0],bounds[:,1])
        absolute_out=out.with_name(out.stem+"_anchored.npz")
        np.savez_compressed(absolute_out,joints=path,rate=rate,side=args.side,start_joints=start)
        print("Measured start degrees:",np.rad2deg(start))
        print("Planned joint min degrees:",np.rad2deg(path.min(0)))
        print("Planned joint max degrees:",np.rad2deg(path.max(0)))
        print("Absolute plan saved:",absolute_out)
        token="START "+args.side+" "+args.mode
        print("Verify hand is EMPTY, starting pose is clear of table/other arm, and local stop is available.")
        print("No automatic Cartesian clearance or inter-arm check is provided. This is commissioning, not hammer replay.")
        input("Press Enter to refresh feedback before final confirmation: ")
        executor.spin_once(timeout_sec=0.)
        # Display and bind to the SAME measured start; never silently re-anchor after approval.
        verify_stack();verify_routes()
        if np.max(np.abs(actual()-start))>.003:raise RuntimeError("Robot moved since plan; rerun to review a new plan")
        answer=input("Type '"+token+"' to send this plan, anything else cancels: ").strip()
        if answer!=token:
            reason="cancelled";return
        for _ in range(10):executor.spin_once(timeout_sec=.01)
        verify_stack();verify_routes()
        if np.max(np.abs(actual()-start))>.003:raise RuntimeError("Start pose changed; no command sent")
        publisher=node.create_publisher(JointState,topic,10)
        until=time.monotonic()+3.
        while publisher.get_subscription_count()!=1 and time.monotonic()<until:executor.spin_once(timeout_sec=.02)
        if publisher.get_subscription_count()!=1:raise RuntimeError("Command subscriber unavailable")
        threading.Thread(target=keyboard,daemon=True).start()
        print("Running small single-arm test. q + Enter or Ctrl+C stops; no automatic return after interruption.")
        armed=True;began=time.monotonic();next_tick=began;last_target=start.copy();last_sent=began
        last_routes=began
        log=out.with_suffix(".csv")
        with log.open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f);writer.writerow(["elapsed_s","error_rad"]+["target_"+str(i) for i in range(6)]+["actual_"+str(i) for i in range(6)])
            duration=(len(path)-1)/rate;settled_since=None
            while True:
                executor.spin_once(timeout_sec=0.)
                now=time.monotonic()
                if not controls.empty():reason="user stop";break
                if now-next_tick>.1:raise RuntimeError("Missed deadline; no catch-up burst")
                current=actual()
                if now-last_routes>.5:verify_routes();last_routes=now
                progress=min((now-began)*rate,len(path)-1);lo=int(progress);hi=min(lo+1,len(path)-1)
                target=path[lo]*(1-(progress-lo))+path[hi]*(progress-lo)
                if np.max(np.abs(target-start))>.02+1e-8:raise RuntimeError("Excursion cap exceeded")
                if np.max(np.abs(target-last_target))/max(now-last_sent,1/rate)>.01:raise RuntimeError("Command velocity cap exceeded")
                error=float(np.max(np.abs(target-current)))
                if error>.03:raise RuntimeError("Tracking error >0.03rad; stopping")
                send(target);last_target=target.copy();last_sent=now
                writer.writerow([now-began,error,*target,*current]);f.flush()
                if now-began>=duration:
                    settled_since=now if error<.003 and settled_since is None else (settled_since if error<.003 else None)
                    if settled_since is not None and now-settled_since>.5:reason="completed and returned to start";break
                    if now-began>duration+8.:raise RuntimeError("Return did not settle")
                next_tick=now+1/rate
                time.sleep(max(0.,next_tick-time.monotonic()))
    except KeyboardInterrupt:
        reason="Ctrl+C"
    except Exception:
        reason="error"
        raise
    finally:
        try:
            if armed:
                try:send(actual())
                except Exception:print("Fresh feedback unavailable: ceased publishing; driver timeout/stop behavior applies.")
                # Give a measured hold one transport cycle. This is not a certified stop.
                executor.spin_once(timeout_sec=.03)
        finally:
            executor.shutdown();node.destroy_node();context.try_shutdown()
            print("Stopped:",reason)


if __name__=="__main__":
    main()
