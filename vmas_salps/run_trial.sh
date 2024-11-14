BATCH=${1}
EXP_NAME=${2}
TRIAL_ID=${3}
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python3 ${SCRIPT_DIR}/run_ccea.py --batch ${BATCH}  --name ${EXP_NAME} --trial_id ${TRIAL_ID}
echo "Finished Trial #${TRIAL_ID}"