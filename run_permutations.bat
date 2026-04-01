@echo off
REM Activate conda environment
call conda activate ptenv


REM Permutation 1: 3 2 0 4 1
echo Processing permutation: 3 2 0 4 1
echo Step 1: Running kfoldcv.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\MOC-AttnSleep"
python kfoldcv.py --device 0 --order "3 2 0 4 1" --np_data_dir "C:\Users\athul\myfiles\Work\Research\Sleep Stage\Data\SHHS"
echo Step 2: Running hmm_input.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\hmm_weighted"
set "saved_path=C:\Users\athul\myfiles\Work\Research\Sleep Stage\SHHS Cascading Order Analysis Result Construction\saved_SHHS\3_2_0_4_1\Combined"
if not exist "%saved_path%" (
    echo Directory %saved_path% does not exist
    exit /b 1
)
python hmm_input.py "%saved_path%" "0.0 0.0 0.0 0.0 0.0" "yes" "val"
echo Completed permutation: 3 2 0 4 1

REM Permutation 1: 3 0 2 4 1
echo Processing permutation: 0 2 3 4 1
echo Step 1: Running kfoldcv.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\MOC-AttnSleep"
python kfoldcv.py --device 0 --order "0 2 3 4 1" --np_data_dir "C:\Users\athul\myfiles\Work\Research\Sleep Stage\Data\SHHS"
echo Step 2: Running hmm_input.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\hmm_weighted"
set "saved_path=C:\Users\athul\myfiles\Work\Research\Sleep Stage\SHHS Cascading Order Analysis Result Construction\saved_SHHS\0_2_3_4_1\Combined"
if not exist "%saved_path%" (
    echo Directory %saved_path% does not exist
    exit /b 1
)
python hmm_input.py "%saved_path%" "0.0 0.0 0.0 0.0 0.0" "yes" "val"
echo Completed permutation: 0 2 3 4 1

REM Permutation 1: 3 0 2 4 1
echo Processing permutation: 0 3 2 4 1
echo Step 1: Running kfoldcv.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\MOC-AttnSleep"
python kfoldcv.py --device 0 --order "0 3 2 4 1" --np_data_dir "C:\Users\athul\myfiles\Work\Research\Sleep Stage\Data\SHHS"
echo Step 2: Running hmm_input.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\hmm_weighted"
set "saved_path=C:\Users\athul\myfiles\Work\Research\Sleep Stage\SHHS Cascading Order Analysis Result Construction\saved_SHHS\0_3_2_4_1\Combined"
if not exist "%saved_path%" (
    echo Directory %saved_path% does not exist
    exit /b 1
)
python hmm_input.py "%saved_path%" "0.0 0.0 0.0 0.0 0.0" "yes" "val"
echo Completed permutation: 0 3 2 4 1

REM Permutation 1: 3 0 2 4 1
echo Processing permutation: 2 3 0 4 1
echo Step 1: Running kfoldcv.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\MOC-AttnSleep"
python kfoldcv.py --device 0 --order "2 3 0 4 1" --np_data_dir "C:\Users\athul\myfiles\Work\Research\Sleep Stage\Data\SHHS"
echo Step 2: Running hmm_input.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\hmm_weighted"
set "saved_path=C:\Users\athul\myfiles\Work\Research\Sleep Stage\SHHS Cascading Order Analysis Result Construction\saved_SHHS\2_3_0_4_1\Combined"
if not exist "%saved_path%" (
    echo Directory %saved_path% does not exist
    exit /b 1
)
python hmm_input.py "%saved_path%" "0.0 0.0 0.0 0.0 0.0" "yes" "val"
echo Completed permutation: 2 3 0 4 1

REM Permutation 1: 3 0 2 4 1
echo Processing permutation: 2 0 3 4 1
echo Step 1: Running kfoldcv.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\MOC-AttnSleep"
python kfoldcv.py --device 0 --order "2 0 3 4 1" --np_data_dir "C:\Users\athul\myfiles\Work\Research\Sleep Stage\Data\SHHS"
echo Step 2: Running hmm_input.py
cd /d "C:\Users\athul\myfiles\Work\Research\Sleep Stage\ConCasNet\hmm_weighted"
set "saved_path=C:\Users\athul\myfiles\Work\Research\Sleep Stage\SHHS Cascading Order Analysis Result Construction\saved_SHHS\2_0_3_4_1\Combined"
if not exist "%saved_path%" (
    echo Directory %saved_path% does not exist
    exit /b 1
)
python hmm_input.py "%saved_path%" "0.0 0.0 0.0 0.0 0.0" "yes" "val"
echo Completed permutation: 2 0 3 4 1

echo All permutations processed.