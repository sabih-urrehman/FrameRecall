

import os
import json
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class DockerManager:
    


    DOCKER_CODECS = {
        'h265', 'hevc', 'libx265',
        'h264', 'avc', 'libx264',
        'av1', 'libaom-av1',
    }

    def __init__(self, container_name="framerecall-h265", verbose=True):
        self.container_name = container_name
        self.verbose = verbose


        self.docker_cmd = self._find_docker_command()
        self.docker_available = self.docker_cmd is not None
        self.container_ready = False


        self.setup_status = "unknown"
        self.project_root = self._find_project_root()


        if self.docker_available:
            self._check_docker_environment()

        if self.verbose:
            logger.info(self.get_status_message())

    def _find_docker_command(self) -> Optional[str]:
        

        if shutil.which("docker.exe"):
            return "docker.exe"

        elif shutil.which("docker"):
            return "docker"
        return None

    def _find_project_root(self) -> Optional[Path]:
        
        current = Path(__file__).parent
        for _ in range(5):
            if (current / "docker").exists():
                return current
            current = current.parent
        return None

    def _check_docker_environment(self):
        
        try:

            result = subprocess.run([self.docker_cmd, "--version"],
                                    capture_output=True, timeout=5)
            if result.returncode != 0:
                self.setup_status = "docker_not_running"
                return


            result = subprocess.run([self.docker_cmd, "images", "-q", self.container_name],
                                    capture_output=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                self.container_ready = True
                self.setup_status = "ready"
            else:
                self.setup_status = "container_missing"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.setup_status = "docker_error"

    def should_use_docker(self, codec: str) -> bool:
        
        return (codec.lower() in self.DOCKER_CODECS and
                self.docker_available and
                self.container_ready)

    def is_available(self) -> bool:
        
        return self.docker_available and self.container_ready

    def get_status_message(self) -> str:
        
        if self.setup_status == "ready":
            return f"✅ Docker backend ready ({self.container_name})"
        elif self.setup_status == "container_missing":
            return f"⚠️  Docker available but {self.container_name} container missing"
        elif self.setup_status == "docker_not_running":
            return "⚠️  Docker installed but not running"
        elif self.setup_status == "docker_error":
            return "❌ Docker daemon error"
        elif not self.docker_available:
            return "ℹ️  Docker not found - native encoding only"
        else:
            return "⚠️  Docker setup status unclear"

    def ensure_container_ready(self, auto_build=False) -> bool:
        
        if self.container_ready:
            return True

        if not self.docker_available:
            return False

        if self.setup_status != "container_missing":
            return False

        if auto_build:
            return self._build_container()

        if self.verbose:
            logger.warning(f"Container {self.container_name} not found. "
                           f"Run 'make build' or enable auto_build=True")
        return False

    def _build_container(self) -> bool:
        
        if not self.project_root:
            if self.verbose:
                logger.error("Cannot find project root with docker/ directory")
            return False

        try:
            dockerfile_path = self.project_root / "docker"

            cmd = [
                self.docker_cmd, "build",
                "-f", str(dockerfile_path / "Dockerfile"),
                "-t", self.container_name,
                str(dockerfile_path)
            ]

            if self.verbose:
                logger.info(f"Building {self.container_name} container...")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self.container_ready = True
                self.setup_status = "ready"
                if self.verbose:
                    logger.info("Container built successfully")
                return True
            else:
                if self.verbose:
                    logger.error(f"Container build failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            if self.verbose:
                logger.error("Container build timed out")
            return False
        except Exception as e:
            if self.verbose:
                logger.error(f"Container build error: {e}")
            return False

    def _convert_ffmpeg_command_paths(self, cmd: List[str], working_dir: Path) -> List[str]:
        

        docker_cmd = []
        working_dir_str = str(working_dir)

        for arg in cmd:

            if working_dir_str in arg:

                converted_arg = arg.replace(working_dir_str, "/workspace")

                converted_arg = converted_arg.replace("\\", "/")
                docker_cmd.append(converted_arg)

            elif "\\" in arg and not arg.startswith("-"):

                converted_arg = arg.replace("\\", "/")

                if not converted_arg.startswith("/"):
                    converted_arg = f"/workspace/{converted_arg}"
                docker_cmd.append(converted_arg)
            else:
                docker_cmd.append(arg)

        return docker_cmd

    def execute_ffmpeg(self, cmd: List[str], working_dir: Path, output_file: Path,
                       auto_build=True, **kwargs) -> Dict[str, Any]:
        


        if not self.ensure_container_ready(auto_build=auto_build):
            raise RuntimeError(f"Docker container {self.container_name} not available")

        if not self.project_root:
            raise RuntimeError("Cannot find project root for script mounting")

        docker_working_dir = self._convert_path_for_docker(working_dir)
        scripts_dir = self._convert_path_for_docker(self.project_root / "docker" / "scripts")


        output_dir = output_file.parent
        docker_output_dir = self._convert_path_for_docker(output_dir)

        print(f"🐛 DOCKER: {working_dir} → {docker_working_dir}")
        print(f"🐛 OUTPUT: {output_dir} → {docker_output_dir}")


        check_cmd = [self.docker_cmd, "run", "--rm",
                     "-v", f"{docker_working_dir}:/workspace", self.container_name,
                     "find", "/workspace", "-name", "*.png"]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True)
        png_count = len(check_result.stdout.strip().split('\n')) if check_result.stdout.strip() else 0
        print(f"🐛 DOCKER: Found {png_count} PNG files in container")


        docker_cmd = self._convert_ffmpeg_command_paths(cmd, working_dir)

        docker_cmd = [arg.replace('/workspace/output/', '/host_output/') for arg in docker_cmd]

        print(f"🐛 FFMPEG CMD: {' '.join(docker_cmd)}")

        cmd_data = {"command": docker_cmd, "working_dir": "/workspace"}
        container_cmd = ["python3", "/scripts/ffmpeg_executor.py", json.dumps(cmd_data)]


        full_docker_cmd = [
                              self.docker_cmd, "run", "--rm",
                              "-v", f"{docker_working_dir}:/workspace",
                              "-v", f"{docker_output_dir}:/host_output",
                              "-v", f"{scripts_dir}:/scripts",
                              self.container_name
                          ] + container_cmd

        try:
            result = subprocess.run(full_docker_cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode != 0:
                print(f"🐛 FFMPEG ERROR: {result.stderr}")
                raise RuntimeError(f"Docker FFmpeg execution failed: {result.stderr}")


            file_size_mb = output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0
            print(f"🐛 SUCCESS: Output file size: {file_size_mb:.2f} MB")

            return {
                "backend": "docker",
                "container": self.container_name,
                "success": True,
                "file_size_mb": round(file_size_mb, 2),
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker FFmpeg execution timed out")
        except Exception as e:
            raise RuntimeError(f"Docker execution error: {e}")

    def _convert_path_for_docker(self, path: Path) -> str:
        
        path_str = str(path.absolute())


        if self._is_wsl():

            if path_str.startswith('/mnt/c'):
                path_str = path_str.replace('/mnt/c', 'C:')
                path_str = path_str.replace('/', '\\')


        return path_str

    def _is_wsl(self) -> bool:
        
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False

    def _prepare_container_command(self, ffmpeg_cmd: List[str], working_dir: str) -> List[str]:
        

        cmd_data = {
            "command": ffmpeg_cmd,
            "working_dir": "/workspace"
        }


        return [
            "python3", "/scripts/ffmpeg_executor.py",
            json.dumps(cmd_data)
        ]

    def execute_command_directly(self, cmd: List[str], working_dir: Path, **kwargs) -> subprocess.CompletedProcess:
        
        if not self.container_ready:
            raise RuntimeError("Docker container not ready")

        docker_working_dir = self._convert_path_for_docker(working_dir)

        docker_cmd = [
                         self.docker_cmd, "run", "--rm",
                         "-v", f"{docker_working_dir}:/workspace",
                         "-w", "/workspace",
                         self.container_name
                     ] + cmd

        return subprocess.run(docker_cmd, **kwargs)

    def get_container_info(self) -> Dict[str, Any]:
        
        info = {
            "docker_available": self.docker_available,
            "docker_cmd": self.docker_cmd,
            "container_ready": self.container_ready,
            "container_name": self.container_name,
            "setup_status": self.setup_status,
            "project_root": str(self.project_root) if self.project_root else None,
            "is_wsl": self._is_wsl()
        }

        if self.docker_available:
            try:

                result = subprocess.run([self.docker_cmd, "--version"],
                                        capture_output=True, text=True, timeout=5)
                info["docker_version"] = result.stdout.strip() if result.returncode == 0 else "unknown"
            except:
                info["docker_version"] = "unknown"

        return info

    def cleanup(self):
        
        if not self.docker_available:
            return

        try:

            subprocess.run([self.docker_cmd, "rmi", self.container_name],
                           capture_output=True, timeout=30)
        except:
            pass