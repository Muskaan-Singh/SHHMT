#!/usr/bin/perl -I /usr/lib/x86_64-linux-gnu/perl/5.22.1/

#  Copyright (C) 2008-2011 Sivaja Nair and (2008-2016)Amba Kulkarni (ambapradeep@gmail.com)
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either
#  version 2 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

use GDBM_File;


tie(%LEX,GDBM_File,$ARGV[0],GDBM_WRCREAT,0666);


while(<STDIN>) {
  chomp;
  @flds = split(/,/,$_);
 # print "input = ",$_,"\n";
  if(($flds[0] !~ /^%/) && ($flds[1] ne "")) {

     if($LEX{$flds[0]} eq "") {
        $LEX{$flds[0]}  =  $flds[1];
     }else {
        $LEX{$flds[0]}  .= "::". $flds[1];
     }
  }
}
untie(%LEX);

