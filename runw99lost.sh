
#python modelsCollect9WIED99Lost.py --batch_size 750 --test_size 0.85 --epochs 300 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1

#python modelsCollect9W99NolgLost.py --batch_size 1100 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1#训练vanishTime
#python modelsCollect9W99NolgLost.py --batch_size 1100 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1

#没有丢失数据状态下，mlp_cf跟车模型0的训练
python model9w99RTlost.py --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1
python model9w99RTlost.py --batch_size 1100 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1

#没有丢失数据状态下，mlp_reg回归模型1的训练
python model9w99RTlost.py --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1
python model9w99RTlost.py --batch_size 1100 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1

#有有丢失数据状态下，丢失slot的预测模型回归模型2的训练
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 2 --fixdata 0 --trainvalmode 1 --goffset 1

#有有丢失数据状态下，采用已经训练好的模型，先slot预测，*再修补*，最后cf的vanishTime模型3的验证
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1


#有有丢失数据状态下，采用已经训练好的模型，先slot预测，*再修补*，最后cf的vanishTime模型3的验证
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 4 --fixdata 3 --trainvalmode 1 --goffset 1

#有有丢失数据状态下，采用已经训练好的模型，先slot预测，*不修补*，最后cf的vanishTime模型3的验证
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 3 --fixdata 0 --trainvalmode 1 --goffset 1

#有有丢失数据状态下，采用已经训练好的模型，先slot预测，*不修补*，最后回归的vanishTime模型3的验证
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 4 --fixdata 0 --trainvalmode 1 --goffset 1


#用于论文实验，不同的跟车模型的numtype和以及环境参数
python model9w99RTlost.py ----num_types 1 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1


#用于论文实验，C0~C9加2参数的统计
python model9w99RTlost.py ----num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 3 --fixdata 0 --trainvalmode 0 --goffset 1


#用于论文实验，对比fvdm 和 idm
nohup python model9idmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 300 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idm.log & 
##用于论文实验，对比fvdm,现有fvdm，修补数据or不修补数据
nohup python model9idmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 300 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > idmM3F3T1.log & 
nohup python model9idmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 300 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 3 --fixdata 0 --trainvalmode 1 --goffset 1 > idmM3F0T1.log & 


#用于论文实验，对比fvdm 和 idm
nohup python model9fvdmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 300 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdm.log & 

##用于论文实验，对比fvdm,现有fvdm，修补数据or不修补数据
nohup python model9fvdmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 300 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > fvdmM3F3T1.log & 
nohup python model9fvdmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 300 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 3 --fixdata 0 --trainvalmode 1 --goffset 1 > fvdmM3F0T1.log


#numtype测试
nohup bash run_idm_numtype_goffset.sh 1 > run_idm_numtype_goffset1.out 2>&1 &
nohup bash run_fvdm_numtype_goffset.sh 1 > run_fvdm_numtype_goffset1.out 2>&1 &
nohup bash run_w99_goffset0.sh 1 > run_w99_goffset0_1.out 2>&1 &
