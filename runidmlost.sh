#-----------------------------------
#没有丢失数据状态下,idm跟车模型0的训练
#-----------------------------------
#没有丢失数据状态下,idm跟车模型0的训练,家用2060

python model9idmRTlost.py  --num_types 4 --batch_size 1300 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTest1_nc0.log &
python model9fvdmRTlost.py --num_types 4 --batch_size 1300 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmTest1_nc0.log &
python model9w99RTlost.py  --num_types 4 --batch_size 1300 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99Test1_nc0.log &



python model9idmRTlost.py  --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTest1_nc0.log
python model9fvdmRTlost.py --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmTest1_nc0.log
python model9w99RTlost.py  --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99Test1_nc0.log


#for 远程大显存服务器
python model9idmRTlost.py  --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTest1_nc0.log &
python model9fvdmRTlost.py --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmTest1_nc0.log &
python model9w99RTlost.py --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99Test1_nc0.log &
python model9w99RTlost.py --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1 > regTest1_nc0.log &


#---------------------------------
#有丢失数据状态下,不训练模型，直接使用已经训练好的模型，进行vanishTime的验证,不修补数据
#---------------------------------
#家用2060
python model9idmRTlost.py  --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 3 --fixdata 0 --trainvalmode 1 --goffset 1 > ex1_table1_idmM3F0T1.log
python model9fvdmRTlost.py --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 3 --fixdata 0 --trainvalmode 1 --goffset 1 > ex1_table1_fvdmM3F0T1.log
python model9w99RTlost.py  --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 3 --fixdata 0 --trainvalmode 1 --goffset 1 > ex1_table1_w99M3F0T1TS50.log

python model9w99RTlost.py  --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 4 --fixdata 0 --trainvalmode 1 --goffset 1 > ex1_table1_regM3F0T1.log





#---------------------------------
#有丢失数据状态下,不训练模型，直接使用已经训练好的模型，进行vanishTime的验证,修补数据
#---------------------------------
#家用2060
python model9idmRTlost.py  --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > idmM3F0T1.log
python model9fvdmRTlost.py --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > fvdmM3F0T1.log
python model9w99RTlost.py  --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > w99M3F0T1.log
python model9w99RTlost.py  --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 4 --fixdata 3 --trainvalmode 1 --goffset 1 > regM3F1T1.log


##for 远程大显存服务器
python model9idmRTlost.py  --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > idmM3F3T1.log
python model9fvdmRTlost.py --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > fvdmM3F3T1.log
python model9w99RTlost.py  --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1 > w99M3F3T1.log
python model9w99RTlost.py  --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 4 --fixdata 3 --trainvalmode 1 --goffset 1 > regM3F3T1.log




#-----------------------------
#numtype测试
#-----------------------------
#家用2060
nohup python model9fvdmRTlost.py --num_types 1 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT1.log  
nohup python model9fvdmRTlost.py --num_types 2 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT2.log  
nohup python model9fvdmRTlost.py --num_types 3 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT3.log  
nohup python model9fvdmRTlost.py --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT4.log  


nohup python model9idmRTlost.py --num_types 1 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT1.log  
nohup python model9idmRTlost.py --num_types 2 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT2.log  
nohup python model9idmRTlost.py --num_types 3 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT3.log  
nohup python model9idmRTlost.py --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT4.log  


nohup python model9w99RTlost.py --num_types 1 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT1.log  
nohup python model9w99RTlost.py --num_types 2 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT2.log  
nohup python model9w99RTlost.py --num_types 3 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT3.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size 1000 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 7000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT4.log  


##for 远程大显存服务器
nohup python model9fvdmRTlost.py --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT4.log  
nohup python model9fvdmRTlost.py --num_types 3 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT3.log  
nohup python model9fvdmRTlost.py --num_types 2 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT2.log  
nohup python model9fvdmRTlost.py --num_types 1 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > fvdmNT1.log  


nohup python model9idmRTlost.py --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT4.log  
nohup python model9idmRTlost.py --num_types 3 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT3.log  
nohup python model9idmRTlost.py --num_types 2 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT2.log  
nohup python model9idmRTlost.py --num_types 1 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmNT1.log  


nohup python model9w99RTlost.py --num_types 4 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT4.log  
nohup python model9w99RTlost.py --num_types 3 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT3.log  
nohup python model9w99RTlost.py --num_types 2 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT2.log  
nohup python model9w99RTlost.py --num_types 1 --batch_size 3900 --test_size 0.5 --epochs 2000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > w99NT1.log  


#-----------------------------
#testSize测试
#-----------------------------

#for 远程大显存服务器
nohup python model9idmRTlost.py --num_types 4 --batch_size 3900 --test_size 0.50 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTS50.log  
nohup python model9idmRTlost.py --num_types 4 --batch_size 2300 --test_size 0.70 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTS70.log  
nohup python model9idmRTlost.py --num_types 4 --batch_size  790 --test_size 0.90 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTS90.log  
nohup python model9idmRTlost.py --num_types 4 --batch_size  395 --test_size 0.95 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTS95.log  

nohup python model9w99RTlost.py --num_types 4 --batch_size 3900 --test_size 0.50 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1 > modelRegTS50.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size 2300 --test_size 0.70 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1 > modelRegTS70.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size  770 --test_size 0.90 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1 > modelRegTS90.log  
nohup python model9w99RTlost.py --num_types 4 --batch_size  380 --test_size 0.95 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1 > modelRegTS95.log  



#for 20260
nohup python model9idmRTlost.py --num_types 4 --batch_size 1300 --test_size 0.50 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTS50.log  
nohup python model9idmRTlost.py --num_types 4 --batch_size 1100 --test_size 0.70 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTS70.log  
nohup python model9idmRTlost.py --num_types 4 --batch_size  790 --test_size 0.90 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTS90.log  
nohup python model9idmRTlost.py --num_types 4 --batch_size  395 --test_size 0.95 --epochs 700 --lr 0.0005 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1 > idmTS95.log