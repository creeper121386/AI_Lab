#!/bin/expect

set rootName /disk/unique/why/AnimeProject
set timeout 3

proc sendFile {root name} {
    spawn scp -r remote unique@115.156.207.244:$root/$name .
    expect {
        "(yes/no)" { send "yes\r"; exp_continue }
        "password:" { send "unique\r" }
    }
    interact
}

sendFile $rootName model2.py
sendFile $rootName train_DCGAN.py