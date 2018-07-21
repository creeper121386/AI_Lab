#!/usr/bin/expect

set timeout 3                  
spawn ssh unique@115.156.207.244  
expect "*yes/no*"
send "yes\r"
expect "*password*"                
send "unique\r"

expect "unique*"
send "cd /disk/unique/why\r"
interact

#cd /disk/unique/why


