#Needed for SHMT interface
OUTENCODING=$4
if [ $OUTENCODING = "DEV" ] ; then
/home/muskaan/scl/amarakosha/relations.pl $1 $2 $3 $4 $5 | /home/muskaan/scl/converters/wx2utf8.sh
fi
if [ $OUTENCODING = "ROMAN" ] ; then
/home/muskaan/scl/amarakosha/relations.pl $1 $2 $3 $4 $5 | /home/muskaan/scl/converters/wx2utf8roman.out
fi
