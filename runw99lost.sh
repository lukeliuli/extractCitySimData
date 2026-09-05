
#python modelsCollect9WIED99Lost.py --batch_size 750 --test_size 0.85 --epochs 1000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1

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
nohup python model9idmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idm.log & 
##用于论文实验，对比fvdm,现有fvdm，修补数据or不修补数据
nohup python model9idmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > idmM3F3T1.log & 
nohup python model9idmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 3 --fixdata 0 --trainvalmode 1 --goffset 1 > idmM3F0T1.log & 


#用于论文实验，对比fvdm 和 idm
nohup python model9fvdmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdm.log & 

##用于论文实验，对比fvdm,现有fvdm，修补数据or不修补数据
nohup python model9fvdmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > fvdmM3F3T1.log & 
nohup python model9fvdmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 3 --fixdata 0 --trainvalmode 1 --goffset 1 > fvdmM3F0T1.log


#numtype测试

nohup python model9fvdmRTlost.py --num_types 1 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT1.log  
nohup python model9fvdmRTlost.py --num_types 2 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT2.log  
nohup python model9fvdmRTlost.py --num_types 3 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT3.log  
nohup python model9fvdmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT4.log  


nohup python model9idmRTlost.py --num_types 1 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT1.log  
nohup python model9idmRTlost.py --num_types 2 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT2.log  
nohup python model9idmRTlost.py --num_types 3 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT3.log  
nohup python model9idmRTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT4.log  


nohup python model9w99RTlost.py --num_types 1 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT1.log  
nohup python model9w99RTlost.py --num_types 2 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT2.log  
nohup python model9w99RTlost.py --num_types 3 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT3.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT4.log  


#testSize测试

nohup python model9w99RTlost.py --num_types 4 --batch_size 3900 --test_size 0.50 --epochs 1000 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99TS50.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size 2300 --test_size 0.70 --epochs 1000 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99TS70.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size  770 --test_size 0.90 --epochs 1000 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99TS90.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size  380 --test_size 0.95 --epochs 1000 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99TS95.log  

nohup python model9w99RTlost.py --num_types 4 --batch_size 3900 --test_size 0.50 --epochs 1000 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1 > modelRegTS50.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size 2300 --test_size 0.70 --epochs 1000 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1 > modelRegTS70.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size  770 --test_size 0.90 --epochs 1000 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1 > modelRegTS90.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size  380 --test_size 0.95 --epochs 1000 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --`trainvalmode 0 --goffset 1 > modelRegTS95.log  
