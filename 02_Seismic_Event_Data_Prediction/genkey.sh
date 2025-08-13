#!/bin/bash

ssh-keygen -q -t rsa
cd .ssh
cat id_rsa.pub >authorized_keys
chmod 600 authorized_keys
ssh login-1 'hostname'
rm ~/genkey.sh
