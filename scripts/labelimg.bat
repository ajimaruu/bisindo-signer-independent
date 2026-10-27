@echo off
rem Open LabelImg for this project (YOLO format, save dir data\labels_yolo).
cd /d "%~dp0.."
".venv-labelimg\Scripts\python.exe" scripts\labelimg_launch.py
if errorlevel 1 pause
