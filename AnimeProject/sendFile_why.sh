#!/bin/expect

set timeout 3
set remotePath root@47.106.247.165:/AnimeProject
#set remotePath unique@115.156.207.244:/disk/unique/why/AnimeProject

proc sendFile {path name} {
    spawn scp -r remote $path/$name .
    expect {
        "(yes/no)" { send "yes\r"; exp_continue }
        "password:" { send "unique\r" }
    }
    interact
}

sendFile $remotePath model2.py
sendFile $remotePath test.py
sendFile $remotePath model/
#sendFile $remotePath train_DCGAN.py