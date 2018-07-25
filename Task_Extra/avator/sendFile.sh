#!/bin/expect

set fname "train_WGAN-GP.py"
set timeout 3
spawn scp $fname unique@115.156.207.244:/disk/unique/why/avator
expect "(yes/no)" 
send "yes\r"
expect "password:" 
send "unique\r"
interact
