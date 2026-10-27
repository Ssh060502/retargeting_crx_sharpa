"""Sensor input, mapping, output policy, and backend-neutral flow execution.

Public exports load on demand so basic backend imports do not load optional
retargeting solvers, native kinematics libraries or runtime sensor dependencies.
"""
from importlib import import_module

_EXPORTS = {
    'AvpRelativeWristMapper': ('teleoperation.observation_mapping', 'AvpRelativeWristMapper'),
    'BimanualRetargetedFrame': ('teleoperation.bimanual', 'BimanualRetargetedFrame'),
    'BimanualRetargetingPipeline': ('teleoperation.bimanual', 'BimanualRetargetingPipeline'),
    'BatchRetargetFlow': ('teleoperation.flow', 'BatchRetargetFlow'),
    'ExecutionFlow': ('teleoperation.flow', 'ExecutionFlow'),
    'ExecutionStatus': ('teleoperation.types', 'ExecutionStatus'),
    'ExecutionStepResult': ('teleoperation.types', 'ExecutionStepResult'),
    'FlowSummary': ('teleoperation.types', 'FlowSummary'),
    'HandInput': ('teleoperation.inputs', 'HandInput'),
    'HandObservationMapper': ('teleoperation.observation_mapping', 'HandObservationMapper'),
    'IdentityHandObservationMapper': ('teleoperation.observation_mapping', 'IdentityHandObservationMapper'),
    'QposCommandLimiter': ('teleoperation.output', 'QposCommandLimiter'),
    'QposOutputFilter': ('teleoperation.output', 'QposOutputFilter'),
    'RelativeWristMapper': ('teleoperation.observation_mapping', 'RelativeWristMapper'),
    'RetargetedFrameResult': ('teleoperation.types', 'RetargetedFrameResult'),
    'SensorHandSample': ('teleoperation.types', 'SensorHandSample'),
    'BimanualSensorHandSample': ('teleoperation.types', 'BimanualSensorHandSample'),
    'StaticCalibrationMapper': ('teleoperation.observation_mapping', 'StaticCalibrationMapper'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
