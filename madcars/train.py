# #!/usr/bin/env python3

# import argparse
# import subprocess
# import os
# import sys
# from pathlib import Path

# def parse_args():
#     parser = argparse.ArgumentParser(description='LocalRunner for MadCars')
#     parser.add_argument('-b', '--base-num', nargs='+',
#         help='Number of base sessions to play as an array', default=[100])
#     parser.add_argument('-f', '--full-num', nargs='+',
#         help='Number of full sessions to play as an array', default=[10])
#     parser.add_argument('--prod', action='store_true',
#         help='Execute production scenario', default=False)
#     parser.add_argument('--python', default='python3')
#     return parser.parse_args()


# def run(steps, work_dir='.'):
#     """Run \n separated commands"""
#     for step in steps.splitlines():
#         step = step.strip()
#         if not step:
#             continue
#         subprocess.check_call(step, shell=True, cwd=work_dir)
#         sys.stdout.flush()


# def train(py, base_num, full_num):
#     agent = str(Path('players/pytorch_main.py'))
#     runner = str(Path('runners/localrunner.py'))
#     train_run = """
#     {py} -u {r} -f \"{py} -u {a} --train\" -s \"{py} -u {a} --train\" -g {n} {full}
#     """
#     for n in base_num:
#         run(train_run.format(py=py, r=runner, a=agent, n=n, full=''))
#     for n in full_num:
#         run(train_run.format(py=py, r=runner, a=agent, n=n, full='--full'))
#     return 0


# def run_prod(py):
#     agent = str(Path('players/pytorch_main.py'))
#     runner = os.path.realpath(os.environ.get(
#         'ORIGREPO', '$HOME/projects/orig_miniaicups'))
#     runner += 'madcars/Runners/localrunner.py'
#     runner = str(Path(runner))
#     prod_run = """
#     {py} -u {r} -f \"{py} -u {a}\" -s \"{py} -u {a}\"
#     """
#     run(prod_run.format(py=py, r=runner, a=agent))
#     return 0


# def main():
#     args = parse_args()
#     if not args.prod:
#         return train(args.python, args.base_num, args.full_num)
#     else:
#         return run_prod(args.python)
#     return 0


# if __name__ == '__main__':
#     sys.exit(main())
