### CODE STARTS
import collections
import datetime
import itertools
import os
import subprocess

from hyperparameters_config import tables


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def get_run_id():
    filename = "logs/expts.txt"
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts) / 5
    return run_id


other_dependencies = {
    "memory": lambda x: int(x["ngpus"]) * 50 if x["gpu"] in ["m40", "titanx"] else int(x["ngpus"]) * 45,
    "cpus": lambda x: int(x["ngpus"]) * 3
}


top_details = "Training PointNet models for ACD + valid shape classification."
hyperparameters = tables

run_id = int(get_run_id())
key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations = list(itertools.product(*value_hyperparameters))

scripts = []
eval_scripts = []

for combo in combinations:
    # Write the scheduler scripts

    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}

    for k, v in other_dependencies.items():
        combo[k] = v(combo)

    od = collections.OrderedDict(sorted(combo.items()))

    if od["downstream_type"] == "modelnet40":
        script_filename = "run_pretraining_template.sh"
    else:
        script_filename = "run_pretraining_seg_template.sh"

    with open(script_filename, 'r') as f:
        schedule_script = f.read()

    print(f"{script_filename}")
    lower_details = ""
    for k, v in od.items():
        lower_details += "%s = %s, " % (k, str(v))
    # removing last comma and space
    lower_details = lower_details[:-2]

    combo["top_details"] = top_details
    combo["lower_details"] = lower_details
    combo["job_id"] = run_id
    print("Scheduling Job #%d" % run_id)

    for k, v in combo.items():
        if "{%s}" % k in schedule_script:
            schedule_script = schedule_script.replace("{%s}" % k, str(v))

    schedule_script += "\n"

    # Write schedule script
    script_name = 'slurm-schedulers/schedule_%d.sh' % run_id
    with open(script_name, 'w') as f:
        f.write(schedule_script)

    scripts.append(script_name)

    # Making files executable
    subprocess.check_output('chmod +x %s' % script_name, shell=True)

    # Update experiment logs
    output = "Script Name = " + script_name + "\n" + \
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" + \
        top_details + "\n" + \
        lower_details + "\n\n"
    with open("logs/expts.txt", "a") as f:
        f.write(output)
    # For the next job
    run_id += 1


# schedule jobs
for script in scripts:
    command = "sbatch %s" % script
    print(subprocess.check_output(command, shell=True))

### CODE ENDS