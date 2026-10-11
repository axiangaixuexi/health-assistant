@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

rem ==================================================================
rem  重要：本文件必须以 UTF-8（带 BOM）+ CRLF 换行保存。
rem  cmd.exe 读取 .bat 时用的是系统 ANSI 代码页（中文 Windows 为 936），
rem  文件若无 BOM，里面的 UTF-8 中文会被按 GBK 解码，所有提示都变成乱码；
rem  仅靠 chcp 65001 修不好，因为 chcp 只改控制台输出代码页，不改 cmd 读文件的方式。
rem ==================================================================

rem 让子进程（python / uv / node）统一按 UTF-8 输出，避免二次乱码
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ==================================================
echo   大健康智能助手 - 一键启动
echo ==================================================
echo.

rem ---------- 1/6 前置依赖 ----------
where uv >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 uv，请先安装：https://docs.astral.sh/uv/
    pause
    exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Node.js，请安装 Node.js 20.19+ 或 22.12+。
    pause
    exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 npm，请先安装 Node.js。
    pause
    exit /b 1
)

rem ---------- 2/6 后端依赖 ----------
if not exist "backend\.venv" (
    echo [后端] 首次运行，正在安装依赖，需要几分钟...
    pushd backend
    call uv sync
    popd
)
if not exist "backend\.venv" (
    echo [错误] 后端依赖安装失败。请手动执行：cd backend 然后 uv sync
    pause
    exit /b 1
)

rem ---------- 3/6 配置文件 ----------
if not exist "backend\.env" (
    copy /y "backend\.env.example" "backend\.env" >nul 2>nul
    echo [提示] 未找到 backend\.env，已从 .env.example 复制一份。
    echo         请填入 API Key（例如 DEEPSEEK_API_KEY），否则聊天会报大模型配置错误。
)

rem ---------- 4/6 RAG 知识库 ----------
if "%SKIP_RAG_INGEST%"=="1" (
    echo [RAG] SKIP_RAG_INGEST=1，跳过知识库构建。
    goto :rag_done
)
if exist "backend\rag\chroma_db\chroma.sqlite3" goto :rag_done
if "%FORCE_RAG_INGEST%"=="1" goto :rag_build
if exist "backend\rag\chroma_db\.ingest_failed" (
    echo [RAG] 上次构建失败，本次已跳过，不再重复尝试。
    echo       修复后重试：删除 backend\rag\chroma_db\.ingest_failed 再运行本 bat，
    echo       或设置 FORCE_RAG_INGEST=1 强制重试。
    goto :rag_done
)

:rag_build
echo [RAG] 正在构建知识库：需从 HuggingFace 下载约 90MB 嵌入模型，请耐心等待...
uv run --project backend python -m backend.rag.ingest
if errorlevel 1 goto :rag_failed
echo [RAG] 知识库构建完成。
goto :rag_done

:rag_failed
if not exist "backend\rag\chroma_db" mkdir "backend\rag\chroma_db"
echo failed> "backend\rag\chroma_db\.ingest_failed"
echo.
echo [警告] 知识库构建失败。后端仍会启动，但 RAG 检索为空，聊天时可能直接报错。
echo.
echo   常见原因：huggingface.co 无法访问（国内网络被墙），报错里会出现
echo              could not connect to huggingface.co 或 bge-small-zh-v1.5。
echo   解决办法：
echo     1. 在 backend\.env 末尾加一行：HF_ENDPOINT=https://hf-mirror.com
echo     2. 删除 backend\rag\chroma_db\.ingest_failed
echo     3. 重新运行本 bat
echo.
echo   已写入失败标记，下次启动不会重复尝试。

:rag_done

rem ---------- 5/6 前端依赖 ----------
if not exist "frontend\node_modules" (
    echo [前端] 首次运行，正在安装依赖，需要几分钟...
    pushd frontend
    call npm install
    popd
)
if not exist "frontend\node_modules" (
    echo [错误] 前端依赖安装失败。请手动执行：cd frontend 然后 npm install
    pause
    exit /b 1
)

rem ---------- 6/6 端口占用检查 ----------
set "PID_8000="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":8000 .*LISTENING"') do set "PID_8000=%%P"
if defined PID_8000 (
    echo [警告] 端口 8000 已被占用，进程 PID = %PID_8000%，后端可能启动失败。
    echo        请先结束该进程，或改用其他端口。
)
set "PID_5173="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":5173 .*LISTENING"') do set "PID_5173=%%P"
if defined PID_5173 (
    echo [警告] 端口 5173 已被占用，进程 PID = %PID_5173%，Vite 会自动改用下一个端口。
)

echo.
echo [启动] 后端 http://localhost:8000    前端 http://localhost:5173
echo.
start "大健康-后端" cmd /k "chcp 65001 >nul && uv run --project backend uvicorn backend.main:app --reload --port 8000"
start "大健康-前端" cmd /k "chcp 65001 >nul && cd frontend && npm run dev"

timeout /t 6 >nul

where curl >nul 2>nul
if errorlevel 1 goto :skip_health
curl -s -o NUL --max-time 3 http://127.0.0.1:8000/health
if errorlevel 1 (
    echo [警告] 后端 6 秒内未响应 /health，请查看「大健康-后端」窗口里的报错。
) else (
    echo [OK]   后端已就绪。
)

:skip_health
start "" http://localhost:5173

echo.
echo 两个服务窗口已启动：
echo   后端 http://localhost:8000     接口文档 http://localhost:8000/docs
echo   前端 http://localhost:5173
echo 关闭对应的 cmd 窗口即可停止服务。
echo.
pause
