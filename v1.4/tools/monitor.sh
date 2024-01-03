#! /bin/sh
if [ -f "./free.log_"`hostname` ];then
    echo "file existed , now deleting it !"
    rm ./free.log_`hostname`
fi
echo "start recording!"

while true
do
    echo "******["`date +%Y-%m-%d_%H:%M:%S`"]******" >> free.log_`hostname`
    free -s 1 -c 2 -h|sed -n 1,2p >> free.log_`hostname`
done