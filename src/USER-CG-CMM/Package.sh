# Update package files in LAMMPS
# cp package file to src if doesn't exist or is different
# do not copy molecular and kspace files if corresponding versions do not exist

for file in *.cpp *.h; do
  if (test $file = angle_cg_cmm.cpp -a ! -e ../pair_angle_harmonic.cpp) then
    continue
  fi
  if (test $file = angle_cg_cmm.h -a ! -e ../pair_angle_harmonic.h) then
    continue
  fi
  if (test $file = ewald_cg.cpp -a ! -e ../ewald.cpp) then
    continue
  fi
  if (test $file = ewald_cg.h -a ! -e ../ewald.h) then
    continue
  fi
  if (test $file = pppm_cg.cpp -a ! -e ../pppm.cpp) then
    continue
  fi
  if (test $file = pppm_cg.h -a ! -e ../pppm.h) then
    continue
  fi
  if (test $file = pair_cg_cmm_coul_long.cpp -a ! -e ../pair_lj_cut_coul_long.cpp) then
    continue
  fi
  if (test $file = pair_cg_cmm_coul_long.h -a ! -e ../pair_lj_cut_coul_long.h) then
    continue
  fi

  if (test ! -e ../$file) then
    echo "  creating src/$file"
    cp $file ..
  elif (test "`diff --brief $file ../$file`" != "") then
    echo "  updating src/$file"
    cp $file ..
  fi
done
