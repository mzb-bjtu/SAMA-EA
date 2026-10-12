"""Convert native RCMP benchmark files to structured JSON."""
import os
import json
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

class RCMPAutoConverter:

    DEFAULT_INPUT = "RCMP/MPLIB1"
    DEFAULT_OUTPUT = "JSON/MPLIB1"

    @staticmethod
    def parse(file_path):

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        ptr = 0
        file_name = os.path.basename(file_path)

        try:

            num_projects = int(lines[ptr])
            ptr += 1
            num_resources = int(lines[ptr])
            ptr += 1
            resource_avail = list(map(int, lines[ptr].split()))
            ptr += 1

            projects = []
            for project_idx in range(num_projects):

                num_acts, release_date = map(int, lines[ptr].split())
                ptr += 1
                resource_flags = list(map(int, lines[ptr].split()))
                ptr += 1

                activities = []
                for act_idx in range(num_acts):
                    parts = lines[ptr].split()
                    ptr += 1

                    duration = int(parts[0])
                    demands = list(map(int, parts[1:1 + num_resources]))
                    num_successors = int(parts[1 + num_resources])

                    successors = []
                    if num_successors > 0:
                        successors = [tuple(map(int, s.split(':')))
                                      for s in parts[2 + num_resources:2 + num_resources + num_successors]]

                    activities.append({
                        'activity_id': f"{project_idx + 1}.{act_idx + 1}",
                        'duration': duration,
                        'resource_demands': demands,
                        'successors': successors
                    })

                projects.append({
                    'project_id': project_idx + 1,
                    'num_activities': num_acts,
                    'release_date': release_date,
                    'resource_flags': resource_flags,
                    'activities': activities
                })

            return {
                'file_name': file_name,
                'num_projects': num_projects,
                'num_resources': num_resources,
                'resource_avail': resource_avail,
                'projects': projects
            }
        except Exception as e:
            raise ValueError(f"解析错误[{file_name}第{ptr + 1}行]: {str(e)}")

    @classmethod
    def convert_file(cls, input_path, output_dir):

        try:
            data = cls.parse(input_path)
            output_path = Path(output_dir) / f"{Path(input_path).stem}.json"

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return (True, f"{Path(input_path).name} → {output_path.name}")
        except Exception as e:
            return (False, f"{Path(input_path).name} ❌ {str(e)}")

    @classmethod
    def batch_convert(cls, parallel=True):

        input_dir = Path.cwd() / cls.DEFAULT_INPUT
        output_dir = Path.cwd() / cls.DEFAULT_OUTPUT

        output_dir.mkdir(exist_ok=True)

        if not input_dir.exists():
            print(f"错误：输入文件夹不存在 {input_dir}")
            print("请创建文件夹并放入.rcmp文件")
            sys.exit(1)

        rcmp_files = sorted(input_dir.glob("*.rcmp"))
        if not rcmp_files:
            print(f"未发现.rcmp文件于 {input_dir}")
            sys.exit(0)

        print(f"▶ 开始转换 {len(rcmp_files)} 个RCMP文件")
        print(f"├─ 输入目录: {input_dir}")
        print(f"└─ 输出目录: {output_dir}\n")

        task_args = [(str(f), str(output_dir)) for f in rcmp_files]

        if parallel and len(rcmp_files) > 1:
            with Pool(min(cpu_count(), len(rcmp_files))) as pool:
                results = pool.starmap(cls.convert_file, task_args)
        else:
            results = [cls.convert_file(*args) for args in task_args]

        success = sum(1 for r in results if r[0])
        print("\n转换结果:")
        for status, msg in results:
            color = "\033[32m" if status else "\033[31m"
            print(f"{color}{msg}\033[0m")

        print(f"\n✅ 成功: {success} 个 | ❌ 失败: {len(results) - success} 个")

        if success < len(results):
            sys.exit(1)

def main():

    print("=" * 50)
    print("RCMP自动转换工具".center(40))
    print("=" * 50)

    try:
        RCMPAutoConverter.batch_convert(parallel=True)
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"致命错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
