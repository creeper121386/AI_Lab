#!/bin/expect

set fname "train_DCGAN.py"
set timeout 3
spawn scp remote unique@115.156.207.244:/disk/unique/why/avator/$fname .
expect "(yes/no)" 
send "yes\r"
expect "password:" 
send "unique\r"
interact
