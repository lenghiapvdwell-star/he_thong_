@echo off
title HE THONG CHUNG KHOAN 2026
color 0A
cls

echo ======================================================
echo    BUOC 1: DON DEP HE THONG
echo ======================================================
if exist "__pycache__" rd /s /q "__pycache__"
echo Don dep xong.

echo.
echo ======================================================
echo    BUOC 2: CAP NHAT GIA MOI NHAT
echo ======================================================
python update_gia.py

echo.
echo ======================================================
echo    BUOC 3: QUET DIEM RU TOAN SAN HOSE
echo ======================================================
python loc_ru_hang.py

echo.
echo ======================================================
echo    BUOC 4: THEO DOI LIVE VA XAC NHAN DIEM MUA
echo ======================================================
python sieu_sao_theo_doi.py

echo.
echo PHAN TICH KET THUC.
pause