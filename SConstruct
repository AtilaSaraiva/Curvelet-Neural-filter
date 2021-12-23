from rsf.proj import *

# Flow("weights/curvFilter_lr_0.01_None.h5","rtmlap-sigsbee.rsf rtmMigModlap-sigsbee.rsf curveletFilter.py experiment.json curvModel.py",
# "python curveletFilter.py m1=${SOURCES[0]} m2=${SOURCES[1]} param=${SOURCES[3]}",stdout=0)

# Flow("filtrado.rsf","rtmlap-sigsbee-muted.rsf weights/curvFilter_lr_0.01_None.h5 experiment.json",
# "python sfcurveletFilterUsage.py m1=${SOURCES[0]} weights=${SOURCES[1]} param=${SOURCES[2]}")

# Flow('filtrado-muted','filtrado sigsbee-mask','''math mask=${SOURCES[1]} output='mask*input' --out=stdout''')

# Flow("filtrado-muted.txt",["filtrado-muted.rsf","rtmlap-sigsbee-muted.rsf"],
# "python maxmin.py m1=${SOURCES[0]} m2=${SOURCES[1]}")

Flow("filtrado-window","filtrado-muted","""
        window f1=426 n2=157 f2=630 n1=164
        """)
        # window f2=362 n2=220 f1=433 n1=150


Flow("rtmlap-muted", "rtmlap","window f1=10")
Flow("rtmMigModlap-muted", "rtmMigModlap","window f1=10")

Flow("weights/curvFilter_lr_0.01_marm.h5","rtmlap-muted.rsf rtmMigModlap-muted.rsf curveletFilter.py experiment-marm.json curvModel.py",
"python curveletFilter.py m1=${SOURCES[0]} m2=${SOURCES[1]} tag=marm param=${SOURCES[3]}",stdout=0)

Flow("filtrado-marm.rsf","rtmlap-muted.rsf weights/curvFilter_lr_0.01_marm.h5 experiment-marm.json",
"python sfcurveletFilterUsage.py m1=${SOURCES[0]} weights=${SOURCES[1]} param=${SOURCES[2]}")

Flow("filtrado-marm.txt",["filtrado-marm.rsf","rtmlap-muted.rsf"],
"python maxmin.py m1=${SOURCES[0]} m2=${SOURCES[1]}")

Flow("filtrado-marm-window","filtrado-marm","""
        window f2=244 n2=88 f1=187 n1=137
        """)

End()
