

```bash
make setup-wsl
make wsl-info
```



**Problem**: `docker: command not found` or `make build` fails

**Solutions**:
1. **Enable WSL Integration in Docker Desktop**:
    - Open Docker Desktop on Windows
    - Go to Settings → Resources → WSL Integration
    - Enable "Enable integration with my default WSL distro"
    - Enable integration with your specific distro (Ubuntu, etc.)
    - Click "Apply & Restart"

2. **Restart WSL** (in Windows Command Prompt):
   ```cmd
   wsl
   wsl
   ```

3. **Check Docker Desktop is running**:
    - Look for Docker whale icon in Windows system tray
    - Should show "Docker Desktop is running"


**Problem**: Encoding is very slow in WSL

**Solutions**:

1. **Use WSL 2** (much faster than WSL 1):
   ```cmd

   wsl
   wsl
   ```

2. **Store files in WSL filesystem**:
   ```bash


   

   cp -r /mnt/c/path/to/framerecall ~/framerecall
   cd ~/framerecall
   ```

3. **Configure WSL memory** (create `C:\Users\YourName\.wslconfig`):
   ```ini
   [wsl2]
   memory=8GB
   processors=4
   swap=2GB
   ```


**Problem**: Container can't see files in `data/` directory

**Solutions**:

1. **Check file permissions**:
   ```bash
   ls -la data/
   chmod 755 data/
   chmod 644 data/input/*
   ```

2. **Use absolute paths**:
   ```bash

   docker run -v "$(pwd)/data:/compute" framerecall-h265 encode input.json output.mp4
   ```

3. **Verify mount is working**:
   ```bash
   docker run
   ```


**Problem**: "Out of memory" or very slow encoding

**Solutions**:

1. **Check available memory**:
   ```bash
   free -h
   make wsl-info
   ```

2. **Use smaller batch sizes** for large datasets:
   ```bash

   split -l 1000 large_chunks.json chunk_
   ```

3. **Increase WSL memory allocation** (see


**Problem**: Windows/Linux path confusion

**Solutions**:

1. **Always use forward slashes** in WSL:
   ```bash

   /home/user/framerecall/data/input/file.json
   

   C:\Users\user\framerecall\data\input\file.json
   ```

2. **Use WSL paths for Docker volumes**:
   ```bash

   make encode INPUT=file.json OUTPUT=video.mp4
   

   docker run -v "$(pwd)/data:/compute" framerecall-h265 encode file.json video.mp4
   ```



1. **Use the optimized command**:
   ```bash
   make encode-large INPUT=big_file.json OUTPUT=big_video.mp4
   ```

2. **Monitor resources**:
   ```bash

   htop

   docker stats
   ```

3. **Use SSD storage** if possible for temp files


| Feature | WSL 1 | WSL 2 |
|---------|-------|-------|
| Docker performance | Slow | Fast |
| File system | Windows FS | Linux FS |
| Memory usage | Shared | Allocated |
| **Recommendation** | Upgrade to WSL 2 | ✅ Use this |


Run this complete test:

```bash
make setup-wsl

make wsl-info

./examples/getting_started.sh

echo '["test chunk '$i'" for i in {1..100}]' | jq -c . > data/input/test_100.json
make encode INPUT=test_100.json OUTPUT=test_100.mp4
```


If you're still having issues:

1. **Check Docker Desktop logs**:
    - Open Docker Desktop → Troubleshoot → View logs

2. **Check WSL logs**:
   ```bash
   dmesg | tail -20
   ```

3. **Verify versions**:
   ```bash
   wsl
   docker
   make setup-wsl
   ```

4. **Test basic Docker functionality**:
   ```bash
   docker run
   docker run
   ```


1. **Use Windows Terminal** for better WSL experience
2. **Install Docker Desktop with WSL 2 backend**
3. **Keep projects in WSL filesystem** (`/home/user/...`)
4. **Use VSCode with WSL extension** for seamless development
5. **Monitor Windows Task Manager** to see Docker Desktop resource usage