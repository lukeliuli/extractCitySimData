#!/bin/bash
GOFFSET="${1:-0}"

# 1) 定义输出文件（按 goffset 区分，方便 0/1 对照）
RESULT_FILE="./tmpModes/w99_goffset_${GOFFSET}_results.txt"

# 2) 表头同时写到屏幕和文件（tee 不带 -a 会覆盖旧文件，正好每次从头写）
{
    printf "%-26s %-8s %-8s\n" "模型类别" "MAE" "RMSE"
    printf "%-26s %-8s %-8s\n" "--------------------------" "--------" "--------"
} | tee "$RESULT_FILE"

for nt in 1 2 3 4; do
    echo ""
    echo ">>> num_types=${nt}, goffset=${GOFFSET} 训练中 ..."
    python model9w99RTlost.py \
        --num_types ${nt} --batch_size 300 --test_size 0.85 \
        --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 \
        --dt 0.1 --nC 2000 --model 0 --fixdata 0 \
        --trainvalmode 0 --goffset ${GOFFSET}

    logfile=$(ls -t ./tmpModes/trainlog_*_1000_0_0_300_0.log 2>/dev/null | head -1)
    if [ -z "$logfile" ]; then
        echo "!! 未找到日志，跳过 num_types=${nt}"
        continue
    fi

    res=$(awk '/Validation Results/ {
        split($0, p, "RMSE: "); split(p[2], r, ","); rmse=r[1]+0
        split($0, q, "MAE: ");   split(q[2], m, ","); mae=m[1]+0
        if (first==0 || mae<bestMae) { first=1; bestMae=mae; bestRmse=rmse }
    } END {
        if (first) printf "%.4f %.4f", bestMae, bestRmse; else printf "NA NA"
    }' "$logfile")

    set -- $res
    # 3) 结果行：屏幕显示 + 追加到文件
    printf "%-26s %-8.4f %-8.4f\n" "${nt}类Wiedemann跟车模型" "$1" "$2" | tee -a "$RESULT_FILE"
done

echo ""
echo "结果已保存到: $RESULT_FILE"