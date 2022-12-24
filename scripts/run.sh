# main
nohup python main.py  --scenario 3m  --total_episode 10000 > runoob1.log 2>&1 &
nohup python main.py  --scenario 8m  --total_episode 10000 > runoob2.log 2>&1 &
nohup python main.py  --scenario 5m_vs_6m  --total_episode 10000  > runoob3.log 2>&1 &
nohup python main.py  --scenario 8m_vs_9m  --total_episode 10000  > runoob4.log 2>&1 &
nohup python main.py  --scenario MMM   --total_episode 10000  > runoob5.log 2>&1 &
nohup python main.py  --scenario 2s3z  --total_episode 10000  > runoob6.log 2>&1 &
nohup python main.py  --scenario 3s5z  --total_episode 10000 > runoob7.log 2>&1 &
