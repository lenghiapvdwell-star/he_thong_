@echo off
title HE THONG CHUNG KHOAN TU DONG - 2026
color 0A
cls

echo ======================================================
echo    BUOC 0: DANG CAP NHAT GIA MOI NHAT TU YAHOO
echo ======================================================
python update_gia.py

echo.
echo ======================================================
echo    BUOC 1: QUET TOAN BO SAN HOSE (TIM DIEM RU)
echo ======================================================
python loc_ru_hang.py

echo.
echo ======================================================
echo    BUOC 2: THEO DOI CHI TIET % CHECKLIST
echo ======================================================
python sieu_sao_theo_doi.py

echo.
echo ======================================================
echo    PHAN TICH KET THUC. 
pause