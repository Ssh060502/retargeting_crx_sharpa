"""Generate a DISPLAY-ONLY first-frame transition from a saved measured start.

No ROS imports or commands. Object is held at its initial table pose; this is
not contact simulation or a collision-free path planner.
"""
import argparse
from pathlib import Path
import numpy as np


def prepare_preview(trajectory, reviewed_start, output):
    if Path(output).resolve() in (Path(trajectory).resolve(), Path(reviewed_start).resolve()):
        raise ValueError("Preview output must not overwrite input files")
    with np.load(trajectory,allow_pickle=False) as source, np.load(reviewed_start,allow_pickle=False) as review:
        original=source["qpos56"]
        start=review["measured_start"]
        target=review["first_target"]
        duration=float(review["transition_seconds"])
        if start.shape!=(56,) or target.shape!=(56,) or not np.isfinite(start).all() or not np.isfinite(target).all() or not np.isfinite(duration) or duration<=0:
            raise ValueError("Invalid saved start/target/duration")
        if original.ndim!=2 or original.shape[1]!=56 or not np.isfinite(original).all():
            raise ValueError("Invalid source qpos56")
        if not np.array_equal(review["trajectory"],original) or not np.array_equal(target,original[0]):
            raise ValueError("Reviewed start belongs to a different trajectory; do not mix old/new table heights")
        count=int(np.ceil(duration*50))+1
        u=np.linspace(0.,1.,count)
        blend=u**3*(10.-15.*u+6.*u**2)
        q=start[None,:]+blend[:,None]*(target-start)[None,:]
        q[0],q[-1]=start,target
        data={"qpos56":q,"fps":(count-1)/duration,"sides":source["sides"],
              "left_base":source["left_base"],"right_base":source["right_base"],
              "preview_only":True,"preview_description":"Measured start to first target; object fixed at initial pose; no collision certification",
              "transition_seconds":duration}
        for name in ("table_height","source_table_z","scene_transform"):
            if name in source:data[name]=source[name]
        for name in source.files:
            if name.startswith("object_") and name.endswith("_scene_poses"):
                poses=source[name]
                if poses.shape!=(len(original),4,4) or not np.isfinite(poses).all():
                    raise ValueError("Invalid source object poses")
                data[name]=np.repeat(poses[:1],count,axis=0)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**data)
    print("DISPLAY ONLY:",output,"duration s",duration,"frames",count)
    for side,offset in (("left",0),("right",28)):
        print(side,"arm delta degrees",np.rad2deg(target[offset:offset+6]-start[offset:offset+6]))
    print("Object remains at first pose. No physical grasp/contact is simulated; no ROS commands are sent.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory",required=True)
    parser.add_argument("--reviewed-start",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    prepare_preview(args.trajectory,args.reviewed_start,args.out)


if __name__=="__main__":main()
