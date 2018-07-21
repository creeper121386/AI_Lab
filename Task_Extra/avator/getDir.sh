#!/bin/expect

set timeout 3
set rootName /disk/unique/why/avator

proc getFile {root name} {
    spawn scp -r remote unique@115.156.207.244:$root/$name .
    expect {
        "(yes/no)" { send "yes\r"; exp_continue }
        "password:" { send "unique\r" }
    }
    interact
}

getFile $rootName data
getFile $rootName saveImg
getFile $rootName model


