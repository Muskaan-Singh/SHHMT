SHELL = /bin/sh

default:
	cd DBM; make

install:
	mkdir -p /var/www/html/scl/amarakosha; mkdir -p /usr/lib/cgi-bin/scl/amarakosha; cp web_interface/*.html /var/www/html/scl/amarakosha; cp -r web_interface/FAQ /var/www/html/scl/amarakosha/; cp -r web_interface/*.cgi /usr/lib/cgi-bin/scl/amarakosha; cp web_interface/amarakosha_knowledge_structure.pdf /var/www/html/scl/amarakosha; cp web_interface/logouoh.png /var/www/html/scl/amarakosha; cp web_interface/about.html /var/www/html/scl/amarakosha; 
	mkdir -p /tmp/SKT_TEMP/AMARA_DOT; chown www-data:www-data /tmp/SKT_TEMP/AMARA_DOT; cd /var/www/html/scl/amarakosha; ln -sfn /tmp/SKT_TEMP/AMARA_DOT DOT

debian_install:
	mkdir -p /tmp/scl//var/www/html/scl/amarakosha; mkdir -p /tmp//scl/usr/lib/cgi-bin/scl/amarakosha; cp web_interface/*.html /tmp/scl//var/www/html/scl/amarakosha; cp -r web_interface/js /tmp/scl//var/www/html/scl/amarakosha;  cp -r web_interface/FAQ /tmp/scl//var/www/html/scl/amarakosha/; cp -r web_interface/*.cgi /tmp/scl//usr/lib/cgi-bin/scl/amarakosha; cp web_interface/amarakosha_knowledge_structure.pdf /tmp/scl//var/www/html/scl/amarakosha; cp web_interface/logouoh.png /tmp/scl//var/www/html/scl/amarakosha; cp web_interface/about.html /tmp/scl//var/www/html/scl/amarakosha; cp web_interface/*.js /tmp/scl//var/www/html/scl/amarakosha;
	mkdir -p /tmp/SKT_TEMP/AMARA_DOT; chown www-data:www-data /tmp/SKT_TEMP/AMARA_DOT; cd /var/www/html/scl/amarakosha; ln -sf /tmp/SKT_TEMP/AMARA_DOT DOT
