import subprocess
import sys
import json
from pathlib import Path
import os
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def create_slurm_script(job_name, script_path, output_path, sample, error_path, commands, n_jobs, psi_flag, time="01:00:00", partition="short", OUT_PATH="", memory="4G", random_delay=False, array=True):
    with open(script_path, "w") as script_file:
        script_file.write("#!/bin/bash\n")
        if random_delay:
            import random
            # Random delay between 1 and 20 seconds
            rand_int = random.randint(1, 20)
            script_file.write(f"#SBATCH --begin=now+{rand_int}seconds\n")
        script_file.write("#SBATCH --requeue\n")
        script_file.write(f"#SBATCH --job-name={job_name}\n")
        script_file.write(f"#SBATCH --output={output_path}\n")
        script_file.write(f"#SBATCH --error={error_path}\n")
        script_file.write(f"#SBATCH --time={time}\n")
        script_file.write(f"#SBATCH --partition={partition}\n")
        script_file.write(f"#SBATCH --mem={memory}\n")
        if n_jobs > 1:
            script_file.write(f"#SBATCH --array=0-{n_jobs - 1}\n")
            script_file.write("\n")
            script_file.write("echo \"Running on node $HOSTNAME\"\n")
            script_file.write("echo \"Job ID: $SLURM_JOB_ID\"\n")
            if array:
                script_file.write("echo \"Running job $SLURM_ARRAY_TASK_ID\"\n")
                script_file.write("echo \"Array Job ID: $SLURM_ARRAY_JOB_ID\"\n")
                script_file.write("echo \"Array Task ID: $SLURM_ARRAY_TASK_ID\"\n")
        script_file.write("\n")

        if psi_flag:
            if array:
                script_file.write("export TARGET_PATH=/scratch/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}\n")
            else:
                script_file.write("export TARGET_PATH=/scratch/$USER/${SLURM_JOB_ID}\n")
            script_file.write("mkdir -p $TARGET_PATH\n")
            script_file.write("\n")
            script_file.write("\n".join(commands))
            script_file.write("\n")
            script_file.write(f"xrdcp -fr $TARGET_PATH/{sample} root://t3dcachedb03.psi.ch:1094/{OUT_PATH}\n")
            if array:
                script_file.write("rm -rf /scratch/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}\n")
            else:
                script_file.write("rm -rf /scratch/$USER/${SLURM_JOB_ID}\n")
        else:
            script_file.write("\n")
            script_file.write("\n".join(commands))
            script_file.write("\n")


class SLURMVanillaSubmitter:
    """
    A class for submitting jobs via SLURM without Dask, one job per file in a sample list of an analysis.
    All jobs for a given samples are submitted to the same cluster.
    The constructor creates a directory .higgs_dna_vanilla_slurm if it does not exist and another one called .higgs_dna_vanilla_slurm/<analysis_name>.
    The data and time (YMD_HMS) is appended to the end of <analysis_name> to avoid overwriting previous submissions.
    Inside this directory two subdirectories called <inputs> and <jobs> will be created.
    In the former the split JSON files will be stored, in the latter the SLURM related job files will be stored.
    To send each job to a separate cluster then set cluster_per_sample=False in the class constructor.

    Parameters:
        :param analysis_name: Name of the analysis.
        :type analysis_name: str
        :param analysis_dict: Dictionary containing the parameters of the analysis.
        :type analysis_dict: dict
        :param original_analysis_path: Path of the original analysis to be replaced with the new ones.
        :type original_analysis_path: str
        :param sample_dict: Dictionary containing the samples and their respective files.
        :type sample_dict: dict
        :param args_string: String containing the command line arguments.
        :type args_string: str
        :param queue: SLURM queue to submit the job to. Defaults to "standard".
        :type queue: str, optional
        :param time: Time request for the job. Defaults to "12:00:00".
        :type time: str, optional
        :param memory: Memory request for the job. Defaults to "10GB".
        :type memory: str, optional
        :param execution_at_psi_pnfs: Is HiggsDNA executed using the Tier 3 at PSI? Necessary when placing files on the PNFS storage element.
        :type execution_at_psi_pnfs: bool, optional
        :param cluster_per_sample: Option to submit each job from a given sample to the same cluster.
        :type cluster_per_sample: bool, optional
    """

    def __init__(
        self,
        analysis_name,
        analysis_dict,
        original_analysis_path,
        OUT_PATH,
        sample_dict,
        args_string,
        queue="standard",
        time="02:00:00",
        memory="10GB",
        execution_at_psi_pnfs=False,
        cluster_per_sample=True,
    ):
        self.datetime_extension = subprocess.getoutput("date +%Y%m%d_%H%M%S")
        self.analysis_name = f"{analysis_name}_{self.datetime_extension}"
        self.analysis_dict = analysis_dict
        self.sample_dict = sample_dict
        self.args_string = args_string
        self.queue = queue
        self.memory = memory
        self.cluster_per_sample = cluster_per_sample
        self.current_dir = os.getcwd()
        self.base_dir = os.path.join(self.current_dir, ".higgs_dna_vanilla_slurm")
        self.analysis_dir = os.path.join(self.base_dir, self.analysis_name)

        self.input_dir = os.path.join(self.analysis_dir, "inputs")
        Path(self.input_dir).mkdir(parents=True, exist_ok=True)

        # split analysis_dict and sample_dict in different JSON files
        self.json_analysis_files = {}
        self.json_sample_files = {}
        for sample in sample_dict:
            self.json_analysis_files[sample] = []
            self.json_sample_files[sample] = []
            for fl in sample_dict[sample]:
                sample_to_dump = {}
                sample_to_dump[sample] = [fl]
                root_file_name = fl.split("/")[-1].split(".")[0]
                sample_file_name = os.path.join(
                    self.input_dir, f"{sample}-{root_file_name}.json"
                )
                with open(sample_file_name, "w") as jf:
                    json.dump(sample_to_dump, jf, indent=4)
                self.json_sample_files[sample].append(sample_file_name)
                an_file_name = os.path.join(
                    self.input_dir, f"AN-{sample}-{root_file_name}.json"
                )
                an_to_dump = deepcopy(self.analysis_dict)
                an_to_dump["samplejson"] = sample_file_name
                with open(an_file_name, "w") as jf:
                    json.dump(an_to_dump, jf, indent=4)
                self.json_analysis_files[sample].append(an_file_name)

        # create job submission directory
        self.jobs_dir = os.path.join(self.analysis_dir, "jobs")
        Path(self.jobs_dir).mkdir(parents=True, exist_ok=True)
        self.job_files = []

        # write job files if running on single cluster
        if self.cluster_per_sample:
            # Get proxy information (required in executable script for this method of running)
            try:
                stat, out = subprocess.getstatusoutput("voms-proxy-info -e --valid 5:00")
            except:
                logger.exception(
                    "voms proxy not found or validity less that 5 hours:\n%s",
                    out
                )
                raise
            try:
                stat, out = subprocess.getstatusoutput("voms-proxy-info -p")
                out = out.strip().split("\n")[-1]
            except:
                logger.exception(
                    "Unable to voms proxy:\n%s",
                    out
                )
                raise

            for sample in sample_dict:
                base_name = f"AN-{sample}"
                jobs_dir = os.path.realpath(self.jobs_dir)
                # replacing /eos/home- with /eos/user/ to prevent problems with output_destination
                jobs_dir = jobs_dir.replace("/eos/home-", "/eos/user/")
                job_file_executable = os.path.join(jobs_dir, f"{base_name}.sh")
                job_file_out = os.path.join(jobs_dir, f"{base_name}_%A_%a.out")
                job_file_err = os.path.join(jobs_dir, f"{base_name}_%A_%a.err")
                n_jobs = len(self.json_analysis_files[sample])

                command = []
                for i, json_file in enumerate(self.json_analysis_files[sample]):
                    arguments = self.args_string.replace(
                        original_analysis_path, json_file
                    ).replace(" vanilla_slurm" if execution_at_psi_pnfs == False else " vanilla_slurm/psi_pnfs", " iterative")
                    if execution_at_psi_pnfs:
                        # Need to change the output path from the storage element to the scratch space
                        arguments_list = arguments.split(" ")
                        dump_index = arguments_list.index('--dump')
                        dump_path = "$TARGET_PATH"
                        arguments_list[dump_index + 1] = dump_path
                        arguments = " ".join(arguments_list)
                    if self.queue != "short":
                        # Activating automatic requeueing of the jobs in case of failure
                        command.append(
                            f'if [[ "$SLURM_ARRAY_TASK_ID" -eq {i} ]]; then '
                            f'/usr/bin/env {sys.prefix}/bin/run_analysis.py {arguments} || '
                            '{ echo "Script ${i} failed, sleeping for 1h and requeuing job then..."; sleep 1h; scontrol requeue $SLURM_JOB_ID; }; '
                            'fi'
                        )
                    else:
                        # For short queue, we do not requeue the jobs, since the overall runtime is too short
                        command.append(f'if [[ "$SLURM_ARRAY_TASK_ID" -eq {i} ]]; then /usr/bin/env {sys.prefix}/bin/run_analysis.py {arguments}; fi')

                create_slurm_script(
                    job_name=base_name,
                    script_path=job_file_executable,
                    output_path=job_file_out,
                    error_path=job_file_err,
                    commands=command,
                    sample=sample,
                    n_jobs=n_jobs,
                    psi_flag=execution_at_psi_pnfs,
                    time=time,
                    partition=self.queue,
                    OUT_PATH=OUT_PATH,
                    memory=self.memory,
                    array=True,
                )

                self.job_files.append(job_file_executable)

        # write job files for separate clusters
        else:
            for sample in sample_dict:
                for json_file in self.json_analysis_files[sample]:
                    base_name = json_file.split("/")[-1].split(".")[0]
                    jobs_dir = os.path.realpath(self.jobs_dir)
                    # replacing /eos/home- with /eos/user/ to prevent problems with output_destination
                    jobs_dir = jobs_dir.replace("/eos/home-", "/eos/user/")
                    job_file_name = os.path.join(jobs_dir, f"{base_name}.sh")
                    job_file_out = os.path.join(jobs_dir, f"{base_name}.out")
                    job_file_err = os.path.join(jobs_dir, f"{base_name}.err")
                    arguments = self.args_string.replace(
                        original_analysis_path, json_file
                    ).replace(" vanilla_slurm", " iterative")
                    if self.queue != "short":
                        command = [
                            f'/usr/bin/env {sys.prefix}/bin/run_analysis.py {arguments} || '
                            '{ echo "Script failed, sleeping for 1h and requeuing job then..."; sleep 1h; scontrol requeue $SLURM_JOB_ID; }; '
                        ]
                    else:
                        command = [
                            f"/usr/bin/env {sys.prefix}/bin/run_analysis.py {arguments} || exit 107"
                        ]
                    create_slurm_script(
                        job_name=base_name,
                        script_path=job_file_name,
                        output_path=job_file_out,
                        error_path=job_file_err,
                        commands=command,
                        sample=sample,
                        n_jobs=1,
                        psi_flag=execution_at_psi_pnfs,
                        time=time,
                        partition=self.queue,
                        OUT_PATH=OUT_PATH,
                        memory=self.memory,
                        array=False,
                    )

                    self.job_files.append(job_file_name)

    def submit(self):
        """
        A method to submit all the jobs in the jobs_dir to the cluster
        """
        for jf in self.job_files:
            subprocess.run(["sbatch", jf])
        return None
