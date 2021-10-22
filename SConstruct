from rsf.proj import *

Flow("weights/curvFilter_lr_0.01_None.h5","rtmlap-sigsbee.rsf rtmMigModlap-sigsbee.rsf curveletFilter.py experiment.json curvModel.py",
"python curveletFilter.py m1=${SOURCES[0]} m2=${SOURCES[1]} param=${SOURCES[3]}",stdout=0)

Flow("filtrado.rsf","rtmlap-sigsbee-muted.rsf weights/curvFilter_lr_0.01_None.h5 experiment.json",
"python sfcurveletFilterUsage.py m1=${SOURCES[0]} weights=${SOURCES[1]} param=${SOURCES[2]}")

# Flow("filtradoLap","filtrado.rsf","laplace")

# Result("filtrado.rsf","grey gainpanel=a mean=y")


End()
