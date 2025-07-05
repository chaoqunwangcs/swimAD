#!/bin/bash
source ~/.bashrc
conda activate swimAD


# 设置检查频率（秒）
check_interval=1

# 定义一个函数来检查显卡状态
check_gpus() {
    # 获取所有显卡的显存信息
    gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    # 初始化一个数组来存储满足条件的显卡ID
    eligible_gpus=()

    # 遍历每行显卡信息
    while IFS=, read -r gpu_id memory_free; do
        # 检查显存是否大于20GB
        if (( memory_free > 20000 )); then
            eligible_gpus+=("$gpu_id")
        fi
    done <<< "$gpu_info"

    # 检查是否有至少两张显卡满足条件
    if (( ${#eligible_gpus[@]} >= 2 )); then
        # 如果有超过两张显卡满足条件，只选择前两张
        selected_gpus=("${eligible_gpus[@]:0:2}")
        echo "找到满足条件的显卡ID：${eligible_gpus[@]}"
        echo "选择的显卡ID：${selected_gpus[@]}"

        python train.py --cfg cfgs/model_yolo11l_v20250630_0623_0604.yaml --ckpt ../ckpts/yolo11s.pt --name 11S_v0630_0623_0604+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250630_0623_0604.yaml --ckpt ../ckpts/yolo11m.pt --name 11M_v0630_0623_0604+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250630_0623_0604.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0630_0623_0604+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64

        python train.py --cfg cfgs/model_yolo11l_v20250630_0623.yaml --ckpt ../ckpts/yolo11s.pt --name 11S_v0630_0623+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250630_0623.yaml --ckpt ../ckpts/yolo11m.pt --name 11M_v0630_0623+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250630_0623.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0630_0623+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64

        python train.py --cfg cfgs/model_yolo11l_v20250630.yaml --ckpt ../ckpts/yolo11s.pt --name 11S_v0630+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250630.yaml --ckpt ../ckpts/yolo11m.pt --name 11M_v0630+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250630.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0630+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64

        python train.py --cfg cfgs/model_yolo11l_v20250630_shuffle.yaml --ckpt ../ckpts/yolo11s.pt --name 11S_v0630+0630shuffle_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250630_shuffle.yaml --ckpt ../ckpts/yolo11m.pt --name 11M_v0630+0630shuffle_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250630_shuffle.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0630+0630shuffle_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64

        python train.py --cfg cfgs/model_yolo11l_v20250604.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0604+0604_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250604_shuffle.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0604+0604shuffle_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64
        python train.py --cfg cfgs/model_yolo11l_v20250604_no_domain_gap.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0604+0604noDomainGap_bs64 --device "$(IFS=,; echo "${selected_gpus[*]}")" --batch 64

        return 0
    else
        echo "没有足够的显卡满足条件（至少需要两张显存大于20GB的显卡）。"
        return 1
    fi
}

# 主循环，每隔一定时间检查一次
while true; do
    if check_gpus; then
        echo "Python 脚本已启动，退出监控循环。"
        break
    fi
    echo "等待 $check_interval 秒后重新检查..."
    sleep $check_interval
done