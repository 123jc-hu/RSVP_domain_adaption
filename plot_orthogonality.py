# -*- coding: utf-8 -*-
# === Effectiveness of Orthogonal Constraint: THU (Single/Cross) ===
# 生成 2x2 图：BA vs K（单、跨）+ Unique ratio vs K（单、跨），TNNLS 风格
import re
import numpy as np
import matplotlib.pyplot as plt

# ============ 1) 把你的原始结果粘到这里（仅需改动这三段）============
RAW_SINGLE_LICS = r"""
K=1 {'AUC': '0.8255+/-0.0947', 'BA': '0.7562+/-0.0835', ...} total_ratio=1.0000
K=2 {'AUC': '0.8754+/-0.0755', 'BA': '0.8069+/-0.0747', ...} total_ratio=1.0000
K=4 {'AUC': '0.8919+/-0.0702', 'BA': '0.8235+/-0.0731', ...} total_ratio=1.0000
K=6 {'AUC': '0.9006+/-0.0662', 'BA': '0.8326+/-0.0689', ...} total_ratio=1.0000
K=8 {'AUC': '0.9060+/-0.0663', 'BA': '0.8404+/-0.0735', ...} total_ratio=1.0000
K=10 {'AUC': '0.9124+/-0.0645', 'BA': '0.8474+/-0.0718', ...} total_ratio=1.0000
K=12 {'AUC': '0.9152+/-0.0554', 'BA': '0.8518+/-0.0639', ...} total_ratio=1.0000
K=14 {'AUC': '0.9158+/-0.0602', 'BA': '0.8520+/-0.0678', ...} total_ratio=1.0000
K=16 {'AUC': '0.9181+/-0.0610', 'BA': '0.8534+/-0.0680', ...} total_ratio=0.9951
K=18 {'AUC': '0.9176+/-0.0599', 'BA': '0.8537+/-0.0674', ...} total_ratio=1.0000
K=20 {'AUC': '0.9231+/-0.0510', 'BA': '0.8595+/-0.0604', ...} total_ratio=1.0000
K=22 {'AUC': '0.9251+/-0.0506', 'BA': '0.8623+/-0.0605', ...} total_ratio=1.0000
K=24 {'AUC': '0.9190+/-0.0586', 'BA': '0.8551+/-0.0678', ...} total_ratio=0.9967
K=26 {'AUC': '0.9260+/-0.0531', 'BA': '0.8637+/-0.0614', ...} total_ratio=1.0000
K=28 {'AUC': '0.9258+/-0.0526', 'BA': '0.8636+/-0.0618', ...} total_ratio=1.0000
K=30 {'AUC': '0.9284+/-0.0550', 'BA': '0.8667+/-0.0622', ...} total_ratio=0.9917
K=32 {'AUC': '0.9294+/-0.0481', 'BA': '0.8685+/-0.0560', ...} total_ratio=0.9985
K=34 {'AUC': '0.9267+/-0.0537', 'BA': '0.8629+/-0.0653', ...} total_ratio=0.9963
K=36 {'AUC': '0.9237+/-0.0574', 'BA': '0.8599+/-0.0643', ...} total_ratio=0.9948
K=38 {'AUC': '0.9301+/-0.0491', 'BA': '0.8693+/-0.0613', ...} total_ratio=0.9951
K=40 {'AUC': '0.9286+/-0.0510', 'BA': '0.8661+/-0.0617', ...} total_ratio=0.9930
K=42 {'AUC': '0.9282+/-0.0520', 'BA': '0.8637+/-0.0611', ...} total_ratio=0.9914
K=44 {'AUC': '0.9273+/-0.0522', 'BA': '0.8636+/-0.0625', ...} total_ratio=0.9837
K=46 {'AUC': '0.9237+/-0.0588', 'BA': '0.8591+/-0.0686', ...} total_ratio=0.9901
K=48 {'AUC': '0.9276+/-0.0550', 'BA': '0.8644+/-0.0640', ...} total_ratio=0.9782
K=50 {'AUC': '0.9266+/-0.0562', 'BA': '0.8627+/-0.0641', ...} total_ratio=0.9750
K=52 {'AUC': '0.9285+/-0.0523', 'BA': '0.8655+/-0.0618', ...} total_ratio=0.9712
K=54 {'AUC': '0.9280+/-0.0529', 'BA': '0.8651+/-0.0636', ...} total_ratio=0.9580
K=56 {'AUC': '0.9244+/-0.0532', 'BA': '0.8589+/-0.0630', ...} total_ratio=0.9481
K=58 {'AUC': '0.9277+/-0.0526', 'BA': '0.8642+/-0.0638', ...} total_ratio=0.9335
K=60 {'AUC': '0.9309+/-0.0495', 'BA': '0.8676+/-0.0610', ...} total_ratio=0.9292
"""

RAW_SINGLE_GS = r"""
k=1 {'AUC': '0.8156+/-0.0986', 'BA': '0.7507+/-0.0842', ...} total_ratio=1.0000
k=2 {'AUC': '0.8624+/-0.0870', 'BA': '0.7921+/-0.0844', ...}
k=4 {'AUC': '0.8837+/-0.0692', 'BA': '0.8109+/-0.0722', ...} total_ratio=0.9297
k=6 {'AUC': '0.8871+/-0.0685', 'BA': '0.8151+/-0.0706', ...} total_ratio=0.8854
k=8 {'AUC': '0.8969+/-0.0659', 'BA': '0.8217+/-0.0705', ...} total_ratio=0.8555
k=10 {'AUC': '0.9000+/-0.0651', 'BA': '0.8280+/-0.0690', ...} total_ratio=0.8141
k=12 {'AUC': '0.9006+/-0.0628', 'BA': '0.8286+/-0.0685', ...} total_ratio=0.7591
k=14 {'AUC': '0.8980+/-0.0661', 'BA': '0.8272+/-0.0718', ...} total_ratio=0.7600
k=16 {'AUC': '0.8971+/-0.0750', 'BA': '0.8257+/-0.0771', ...} total_ratio=0.7412
k=18 {'AUC': '0.8939+/-0.0701', 'BA': '0.8220+/-0.0719', ...} total_ratio=0.7083
k=20 {'AUC': '0.9031+/-0.0622', 'BA': '0.8325+/-0.0713', ...} total_ratio=0.6852
k=22 {'AUC': '0.9104+/-0.0574', 'BA': '0.8406+/-0.0638', ...} total_ratio=0.6726
k=24 {'AUC': '0.9023+/-0.0594', 'BA': '0.8272+/-0.0624', ...} total_ratio=0.6523
k=26 {'AUC': '0.9063+/-0.0662', 'BA': '0.8374+/-0.0712', ...} total_ratio=0.6106
k=28 {'AUC': '0.9074+/-0.0583', 'BA': '0.8355+/-0.0672', ...} total_ratio=0.6027
k=30 {'AUC': '0.9112+/-0.0539', 'BA': '0.8407+/-0.0614', ...} total_ratio=0.5839
k=32 {'AUC': '0.9118+/-0.0567', 'BA': '0.8425+/-0.0637', ...} total_ratio=0.5732
k=34 {'AUC': '0.9153+/-0.0567', 'BA': '0.8467+/-0.0632', ...} total_ratio=0.5643
k=36 {'AUC': '0.9049+/-0.0630', 'BA': '0.8325+/-0.0690', ...} total_ratio=0.5569
k=38 {'AUC': '0.9140+/-0.0565', 'BA': '0.8460+/-0.0629', ...} total_ratio=0.5350
k=40 {'AUC': '0.9149+/-0.0584', 'BA': '0.8446+/-0.0659', ...} total_ratio=0.5297
k=42 {'AUC': '0.9095+/-0.0613', 'BA': '0.8381+/-0.0678', ...} total_ratio=0.5242
k=44 {'AUC': '0.9114+/-0.0592', 'BA': '0.8409+/-0.0646', ...} total_ratio=0.4883
k=46 {'AUC': '0.9116+/-0.0597', 'BA': '0.8439+/-0.0658', ...} total_ratio=0.5126
k=48 {'AUC': '0.9132+/-0.0573', 'BA': '0.8443+/-0.0679', ...} total_ratio=0.4857
k=50 {'AUC': '0.9124+/-0.0581', 'BA': '0.8444+/-0.0668', ...} total_ratio=0.5103
k=52 {'AUC': '0.9136+/-0.0585', 'BA': '0.8448+/-0.0665', ...} total_ratio=0.4712
k=54 {'AUC': '0.9154+/-0.0553', 'BA': '0.8470+/-0.0614', ...} total_ratio=0.4693
k=56 {'AUC': '0.9117+/-0.0555', 'BA': '0.8399+/-0.0616', ...} total_ratio=0.4713
k=58 {'AUC': '0.9118+/-0.0579', 'BA': '0.8414+/-0.0670', ...} total_ratio=0.4469
k=60 {'AUC': '0.9173+/-0.0555', 'BA': '0.8473+/-0.0622', ...} total_ratio=0.4383
"""

RAW_SINGLE_REG = r"""
k=1 {'AUC': '0.8232+/-0.0885', 'BA': '0.7568+/-0.0773', ...} total_ratio=1.0000
k=2 {'AUC': '0.8655+/-0.0852', 'BA': '0.7968+/-0.0804', ...} total_ratio=1.0000
k=4 {'AUC': '0.8893+/-0.0666', 'BA': '0.8187+/-0.0686', ...} total_ratio=0.9453
k=6 {'AUC': '0.8844+/-0.0802', 'BA': '0.8151+/-0.0786', ...} total_ratio=0.9193
k=8 {'AUC': '0.8956+/-0.0681', 'BA': '0.8233+/-0.0716', ...} total_ratio=0.8828
k=10 {'AUC': '0.9012+/-0.0632', 'BA': '0.8304+/-0.0685', ...} total_ratio=0.8953
k=12 {'AUC': '0.9038+/-0.0580', 'BA': '0.8339+/-0.0648', ...} total_ratio=0.8607
k=14 {'AUC': '0.9036+/-0.0656', 'BA': '0.8345+/-0.0703', ...} total_ratio=0.8594
k=16 {'AUC': '0.9028+/-0.0640', 'BA': '0.8314+/-0.0702', ...} total_ratio=0.8555
k=18 {'AUC': '0.9009+/-0.0671', 'BA': '0.8317+/-0.0699', ...} total_ratio=0.8490
k=20 {'AUC': '0.9046+/-0.0670', 'BA': '0.8371+/-0.0731', ...} total_ratio=0.8469
k=22 {'AUC': '0.9089+/-0.0585', 'BA': '0.8387+/-0.0646', ...} total_ratio=0.8615
k=24 {'AUC': '0.9054+/-0.0637', 'BA': '0.8350+/-0.0681', ...} total_ratio=0.8815
k=26 {'AUC': '0.9085+/-0.0637', 'BA': '0.8395+/-0.0684', ...} total_ratio=0.8660
k=28 {'AUC': '0.9128+/-0.0577', 'BA': '0.8430+/-0.0629', ...} total_ratio=0.8638
k=30 {'AUC': '0.9117+/-0.0583', 'BA': '0.8445+/-0.0653', ...} total_ratio=0.8609
k=32 {'AUC': '0.9131+/-0.0551', 'BA': '0.8429+/-0.0641', ...} total_ratio=0.8721
k=34 {'AUC': '0.9171+/-0.0526', 'BA': '0.8507+/-0.0596', ...} total_ratio=0.8649
k=36 {'AUC': '0.9087+/-0.0582', 'BA': '0.8378+/-0.0622', ...} total_ratio=0.8624
k=38 {'AUC': '0.9171+/-0.0524', 'BA': '0.8497+/-0.0609', ...} total_ratio=0.8787
k=40 {'AUC': '0.9153+/-0.0621', 'BA': '0.8492+/-0.0672', ...} total_ratio=0.8680
k=42 {'AUC': '0.9137+/-0.0621', 'BA': '0.8455+/-0.0698', ...} total_ratio=0.8761
k=44 {'AUC': '0.9152+/-0.0577', 'BA': '0.8460+/-0.0648', ...} total_ratio=0.8810
k=46 {'AUC': '0.9129+/-0.0618', 'BA': '0.8447+/-0.0676', ...} total_ratio=0.8825
k=48 {'AUC': '0.9177+/-0.0549', 'BA': '0.8514+/-0.0636', ...} total_ratio=0.8630
k=50 {'AUC': '0.9177+/-0.0588', 'BA': '0.8495+/-0.0664', ...} total_ratio=0.8506
k=52 {'AUC': '0.9178+/-0.0536', 'BA': '0.8481+/-0.0619', ...} total_ratio=0.8615
k=54 {'AUC': '0.9156+/-0.0610', 'BA': '0.8467+/-0.0686', ...} total_ratio=0.8397
k=56 {'AUC': '0.9118+/-0.0609', 'BA': '0.8441+/-0.0635', ...} total_ratio=0.8262
k=58 {'AUC': '0.9141+/-0.0652', 'BA': '0.8469+/-0.0704', ...} total_ratio=0.8168
k=60 {'AUC': '0.9165+/-0.0535', 'BA': '0.8473+/-0.0639', ...} total_ratio=0.7674
"""

# —— 跨被试：LiCSNet/GS/RegGS ——
RAW_CROSS_LICS = r"""
k=1 {'AUC': '0.7586+/-0.1218', 'BA': '0.6952+/-0.1055', ...} total_ratio=1.0000
k=2 {'AUC': '0.8425+/-0.0970', 'BA': '0.7664+/-0.0958', ...} total_ratio=1.0000
k=4 {'AUC': '0.8827+/-0.0736', 'BA': '0.8087+/-0.0811', ...} total_ratio=1.0000
k=6 {'AUC': '0.8899+/-0.0725', 'BA': '0.8146+/-0.0814', ...} total_ratio=1.0000
k=8 {'AUC': '0.9039+/-0.0637', 'BA': '0.8317+/-0.0769', ...} total_ratio=1.0000
k=10 {'AUC': '0.9088+/-0.0598', 'BA': '0.8367+/-0.0730', ...} total_ratio=1.0000
k=12 {'AUC': '0.9131+/-0.0572', 'BA': '0.8420+/-0.0750', ...} total_ratio=1.0000
k=14 {'AUC': '0.9162+/-0.0562', 'BA': '0.8449+/-0.0717', ...} total_ratio=1.0000
k=16 {'AUC': '0.9188+/-0.0570', 'BA': '0.8479+/-0.0756', ...} total_ratio=1.0000
k=18 {'AUC': '0.9212+/-0.0515', 'BA': '0.8507+/-0.0681', ...} total_ratio=1.0000
k=20 {'AUC': '0.9199+/-0.0546', 'BA': '0.8467+/-0.0751', ...} total_ratio=1.0000
k=22 {'AUC': '0.9216+/-0.0506', 'BA': '0.8526+/-0.0651', ...} total_ratio=1.0000
k=24 {'AUC': '0.9257+/-0.0474', 'BA': '0.8564+/-0.0626', ...} total_ratio=1.0000
k=26 {'AUC': '0.9242+/-0.0504', 'BA': '0.8542+/-0.0680', ...} total_ratio=1.0000
k=28 {'AUC': '0.9267+/-0.0475', 'BA': '0.8573+/-0.0638', ...} total_ratio=1.0000
k=30 {'AUC': '0.9285+/-0.0467', 'BA': '0.8595+/-0.0656', ...} total_ratio=1.0000
k=32 {'AUC': '0.9255+/-0.0486', 'BA': '0.8570+/-0.0670', ...} total_ratio=1.0000
k=34 {'AUC': '0.9285+/-0.0472', 'BA': '0.8608+/-0.0650', ...} total_ratio=1.0000
k=36 {'AUC': '0.9275+/-0.0474', 'BA': '0.8585+/-0.0661', ...} total_ratio=1.0000
k=38 {'AUC': '0.9270+/-0.0481', 'BA': '0.8576+/-0.0675', ...} total_ratio=1.0000
k=40 {'AUC': '0.9265+/-0.0501', 'BA': '0.8605+/-0.0676', ...} total_ratio=1.0000
k=42 {'AUC': '0.9290+/-0.0486', 'BA': '0.8613+/-0.0672', ...} total_ratio=1.0000
k=44 {'AUC': '0.9285+/-0.0455', 'BA': '0.8606+/-0.0641', ...} total_ratio=1.0000
k=46 {'AUC': '0.9289+/-0.0462', 'BA': '0.8591+/-0.0663', ...} total_ratio=1.0000
k=48 {'AUC': '0.9294+/-0.0460', 'BA': '0.8634+/-0.0666', ...} total_ratio=1.0000
k=50 {'AUC': '0.9309+/-0.0438', 'BA': '0.8649+/-0.0600', ...} total_ratio=1.0000
k=52 {'AUC': '0.9302+/-0.0484', 'BA': '0.8630+/-0.0711', ...} total_ratio=1.0000
k=54 {'AUC': '0.9319+/-0.0453', 'BA': '0.8682+/-0.0618', ...} total_ratio=1.0000
k=56 {'AUC': '0.9301+/-0.0462', 'BA': '0.8625+/-0.0649', ...} total_ratio=1.0000
k=58 {'AUC': '0.9324+/-0.0453', 'BA': '0.8654+/-0.0658', ...} total_ratio=1.0000
k=60 {'AUC': '0.9307+/-0.0447', 'BA': '0.8651+/-0.0648', ...} total_ratio=1.0000
"""

RAW_CROSS_GS = r"""
k=1 {'AUC': '0.7790+/-0.1015', 'BA': '0.7134+/-0.0909', ...} total_ratio=1.0000
k=2 {'AUC': '0.8472+/-0.0914', 'BA': '0.7733+/-0.0864', ...} total_ratio=1.0000
k=4 {'AUC': '0.8675+/-0.0831', 'BA': '0.7914+/-0.0890', ...} total_ratio=1.0000
k=6 {'AUC': '0.8858+/-0.0777', 'BA': '0.8126+/-0.0872', ...} total_ratio=0.9818
k=8 {'AUC': '0.8940+/-0.0706', 'BA': '0.8189+/-0.0829', ...} total_ratio=0.9180
k=10 {'AUC': '0.9003+/-0.0685', 'BA': '0.8266+/-0.0797', ...} total_ratio=0.8875
k=12 {'AUC': '0.9017+/-0.0643', 'BA': '0.8287+/-0.0774', ...} total_ratio=0.8906
k=14 {'AUC': '0.9077+/-0.0597', 'BA': '0.8342+/-0.0708', ...} total_ratio=0.8750
k=16 {'AUC': '0.9030+/-0.0650', 'BA': '0.8299+/-0.0792', ...} total_ratio=0.8506
k=18 {'AUC': '0.9088+/-0.0619', 'BA': '0.8380+/-0.0759', ...} total_ratio=0.8220
k=20 {'AUC': '0.9091+/-0.0583', 'BA': '0.8365+/-0.0724', ...} total_ratio=0.8117
k=22 {'AUC': '0.9093+/-0.0619', 'BA': '0.8371+/-0.0768', ...} total_ratio=0.7891
k=24 {'AUC': '0.9120+/-0.0579', 'BA': '0.8375+/-0.0759', ...} total_ratio=0.7728
k=26 {'AUC': '0.9116+/-0.0614', 'BA': '0.8385+/-0.0763', ...} total_ratio=0.7650
k=28 {'AUC': '0.9127+/-0.0586', 'BA': '0.8415+/-0.0749', ...} total_ratio=0.7411
k=30 {'AUC': '0.9153+/-0.0555', 'BA': '0.8446+/-0.0723', ...} total_ratio=0.7443
k=32 {'AUC': '0.9151+/-0.0559', 'BA': '0.8440+/-0.0717', ...} total_ratio=0.7222
k=34 {'AUC': '0.9151+/-0.0570', 'BA': '0.8435+/-0.0732', ...} total_ratio=0.7040
k=36 {'AUC': '0.9163+/-0.0563', 'BA': '0.8449+/-0.0747', ...} total_ratio=0.7023
k=38 {'AUC': '0.9172+/-0.0559', 'BA': '0.8447+/-0.0740', ...} total_ratio=0.6850
k=40 {'AUC': '0.9156+/-0.0602', 'BA': '0.8437+/-0.0784', ...} total_ratio=0.6672
k=42 {'AUC': '0.9175+/-0.0556', 'BA': '0.8491+/-0.0700', ...} total_ratio=0.6570
k=44 {'AUC': '0.9193+/-0.0521', 'BA': '0.8478+/-0.0711', ...} total_ratio=0.6541
k=46 {'AUC': '0.9208+/-0.0518', 'BA': '0.8512+/-0.0681', ...} total_ratio=0.6440
k=48 {'AUC': '0.9221+/-0.0507', 'BA': '0.8490+/-0.0698', ...} total_ratio=0.6341
k=50 {'AUC': '0.9170+/-0.0553', 'BA': '0.8464+/-0.0727', ...} total_ratio=0.6138
k=52 {'AUC': '0.9193+/-0.0550', 'BA': '0.8514+/-0.0704', ...} total_ratio=0.5956
k=54 {'AUC': '0.9213+/-0.0549', 'BA': '0.8494+/-0.0710', ...} total_ratio=0.5943
k=56 {'AUC': '0.9225+/-0.0504', 'BA': '0.8520+/-0.0687', ...} total_ratio=0.5910
k=58 {'AUC': '0.9194+/-0.0539', 'BA': '0.8484+/-0.0727', ...} total_ratio=0.5760
k=60 {'AUC': '0.9212+/-0.0492', 'BA': '0.8498+/-0.0685', ...} total_ratio=0.5810
"""

RAW_CROSS_REG = r"""
k=1 {'AUC': '0.7797+/-0.1005', 'BA': '0.7132+/-0.0899', ...} total_ratio=1.0000
k=2 {'AUC': '0.8429+/-0.0923', 'BA': '0.7687+/-0.0894', ...} total_ratio=1.0000
k=4 {'AUC': '0.8685+/-0.0840', 'BA': '0.7960+/-0.0910', ...} total_ratio=1.0000
k=6 {'AUC': '0.8849+/-0.0757', 'BA': '0.8101+/-0.0856', ...} total_ratio=0.9922
k=8 {'AUC': '0.8948+/-0.0699', 'BA': '0.8206+/-0.0828', ...} total_ratio=0.9824
k=10 {'AUC': '0.9013+/-0.0635', 'BA': '0.8267+/-0.0754', ...} total_ratio=0.9703
k=12 {'AUC': '0.9051+/-0.0638', 'BA': '0.8321+/-0.0753', ...} total_ratio=0.9766
k=14 {'AUC': '0.9064+/-0.0608', 'BA': '0.8333+/-0.0754', ...} total_ratio=0.9777
k=16 {'AUC': '0.9065+/-0.0617', 'BA': '0.8333+/-0.0784', ...} total_ratio=0.9648
k=18 {'AUC': '0.9114+/-0.0559', 'BA': '0.8392+/-0.0720', ...} total_ratio=0.9661
k=20 {'AUC': '0.9157+/-0.0569', 'BA': '0.8428+/-0.0712', ...} total_ratio=0.9547
k=22 {'AUC': '0.9125+/-0.0590', 'BA': '0.8413+/-0.0725', ...} total_ratio=0.9553
k=24 {'AUC': '0.9163+/-0.0556', 'BA': '0.8438+/-0.0738', ...} total_ratio=0.9603
k=26 {'AUC': '0.9168+/-0.0564', 'BA': '0.8451+/-0.0721', ...} total_ratio=0.9555
k=28 {'AUC': '0.9178+/-0.0524', 'BA': '0.8455+/-0.0699', ...} total_ratio=0.9598
k=30 {'AUC': '0.9196+/-0.0543', 'BA': '0.8478+/-0.0733', ...} total_ratio=0.9510
k=32 {'AUC': '0.9221+/-0.0504', 'BA': '0.8525+/-0.0660', ...} total_ratio=0.9478
k=34 {'AUC': '0.9198+/-0.0509', 'BA': '0.8472+/-0.0705', ...} total_ratio=0.9540
k=36 {'AUC': '0.9201+/-0.0539', 'BA': '0.8462+/-0.0732', ...} total_ratio=0.9583
k=38 {'AUC': '0.9222+/-0.0503', 'BA': '0.8515+/-0.0702', ...} total_ratio=0.9622
k=40 {'AUC': '0.9242+/-0.0554', 'BA': '0.8541+/-0.0731', ...} total_ratio=0.9652
k=42 {'AUC': '0.9226+/-0.0506', 'BA': '0.8541+/-0.0665', ...} total_ratio=0.9568
k=44 {'AUC': '0.9207+/-0.0510', 'BA': '0.8480+/-0.0715', ...} total_ratio=0.9499
k=46 {'AUC': '0.9264+/-0.0489', 'BA': '0.8560+/-0.0687', ...} total_ratio=0.9694
k=48 {'AUC': '0.9237+/-0.0511', 'BA': '0.8524+/-0.0695', ...} total_ratio=0.9580
k=50 {'AUC': '0.9271+/-0.0489', 'BA': '0.8552+/-0.0693', ...} total_ratio=0.9694
k=52 {'AUC': '0.9270+/-0.0491', 'BA': '0.8547+/-0.0689', ...} total_ratio=0.9660
k=54 {'AUC': '0.9293+/-0.0456', 'BA': '0.8602+/-0.0645', ...} total_ratio=0.9653
k=56 {'AUC': '0.9288+/-0.0466', 'BA': '0.8576+/-0.0647', ...} total_ratio=0.9517
k=58 {'AUC': '0.9293+/-0.0460', 'BA': '0.8627+/-0.0619', ...} total_ratio=0.9421
k=60 {'AUC': '0.9255+/-0.0501', 'BA': '0.8560+/-0.0696', ...} total_ratio=0.8560
"""

# ---- 参考基线：全通道 LiCS-Backbone（不做通道选择）----
BASE_SINGLE_BA_MEAN, BASE_SINGLE_BA_STD = 0.8634, 0.0595
BASE_CROSS_BA_MEAN,  BASE_CROSS_BA_STD  = 0.8653, 0.0654


# ===== 3) 逐行稳健解析 =====
re_k   = re.compile(r"[Kk]\s*=\s*(\d+)")
re_ba  = re.compile(r"[\"']?BA[\"']?\s*:\s*[\"']\s*([\d.]+)\s*(?:\+/-|±)\s*([\d.]+)\s*[\"']")
re_tr1 = re.compile(r"total_ratio\s*=\s*([\d.]+)")
re_tr2 = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")   # 行尾最后一个数字（兜底）

def parse_block_lines(raw: str):
    Ks, BA_mu, BA_sd, UR = [], [], [], []
    for line in raw.splitlines():
        if not line.strip():
            continue
        mk = re_k.search(line)
        mb = re_ba.search(line)
        if not (mk and mb):
            continue
        k  = int(mk.group(1))
        mu = float(mb.group(1))
        sd = float(mb.group(2))

        # total_ratio：优先有前缀的；否则兜底取行尾数字（且这个数字不等于 BA 的均值/方差）
        mtr = re_tr1.search(line)
        if mtr:
            ur = float(mtr.group(1))
        else:
            mlast = re_tr2.search(line)
            if mlast:
                cand = float(mlast.group(1))
                # 简单去歧义：如果行里已经出现过这个 cand（比如 BA 的均值/方差），则放弃
                # 否则作为 unique ratio
                if (f"{cand}" not in line[:mlast.start()]) or (cand not in (mu, sd)):
                    ur = cand
                else:
                    ur = np.nan
            else:
                ur = np.nan

        Ks.append(k); BA_mu.append(mu); BA_sd.append(sd); UR.append(ur)

    Ks     = np.asarray(Ks, dtype=int)
    BA_mu  = np.asarray(BA_mu, dtype=float)
    BA_sd  = np.asarray(BA_sd, dtype=float)
    UR     = np.asarray(UR, dtype=float)

    # 按 K 排序
    if len(Ks):
        idx = np.argsort(Ks)
        Ks, BA_mu, BA_sd, UR = Ks[idx], BA_mu[idx], BA_sd[idx], UR[idx]
    return Ks, BA_mu, BA_sd, UR

# 解析六段
Ks_s_l, BA_s_l, SD_s_l, UR_s_l = parse_block_lines(RAW_SINGLE_LICS)
Ks_s_g, BA_s_g, SD_s_g, UR_s_g = parse_block_lines(RAW_SINGLE_GS)
Ks_s_r, BA_s_r, SD_s_r, UR_s_r = parse_block_lines(RAW_SINGLE_REG)

Ks_c_l, BA_c_l, SD_c_l, UR_c_l = parse_block_lines(RAW_CROSS_LICS)
Ks_c_g, BA_c_g, SD_c_g, UR_c_g = parse_block_lines(RAW_CROSS_GS)
Ks_c_r, BA_c_r, SD_c_r, UR_c_r = parse_block_lines(RAW_CROSS_REG)

# ===== 4) 画图（TNNLS 风格）=====
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
})

COL_LICS = "#1d3557"  # ours
COL_GS   = "#2a9d8f"
COL_REG  = "#e76f51"
COL_BASE = "#6c757d"

def plot_with_band(ax, K, mu, sd, label, color, marker):
    if len(K) == 0: return
    ax.plot(K, mu, label=label, color=color, lw=1.4, marker=marker, ms=3.0)
    ax.fill_between(K, mu - sd, mu + sd, color=color, alpha=0.15, linewidth=0)

def plot_unique(ax, K, ur, label, color, marker):
    if len(K) == 0: return
    K = np.asarray(K); ur = np.asarray(ur, dtype=float)
    m = ~np.isnan(ur)
    if m.sum():
        ax.plot(K[m], ur[m], color=color, lw=1.4, marker=marker, ms=3.0, label=label)

def add_baseline(ax, mean, std, label="Full-channel baseline"):
    # 确保先设 xlim 再画
    xmin, xmax = ax.get_xlim()
    ax.axhline(mean, color=COL_BASE, lw=1.0, ls="--", label=label, alpha=0.9)
    ax.fill_between([xmin, xmax], [mean-std, mean-std], [mean+std, mean+std],
                    color=COL_BASE, alpha=0.10, linewidth=0)

fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.8), sharex='col')
(ax1, ax2), (ax3, ax4) = axs

# (a) BA vs K (Single)
ax1.set_xlim(0, 62); ax1.set_ylim(0.72, 0.90)
ax1.set_xticks([0, 8, 16, 32, 48, 62])
plot_with_band(ax1, Ks_s_l, BA_s_l, SD_s_l, "Orthogonal Softmax", COL_LICS, "o")
plot_with_band(ax1, Ks_s_g, BA_s_g, SD_s_g, "Vanilla Softmax",        COL_GS,   "s")
plot_with_band(ax1, Ks_s_r, BA_s_r, SD_s_r, "Regularized Softmax",    COL_REG,  "d")
add_baseline(ax1, BASE_SINGLE_BA_MEAN, BASE_SINGLE_BA_STD)
ax1.set_xlabel("Number of selected channels (K)")
ax1.set_ylabel("Balanced Accuracy (Within-)")
ax1.grid(True, lw=0.4, alpha=0.5)

# (b) BA vs K (Cross)
ax2.set_xlim(0, 62); ax2.set_ylim(0.72, 0.90)
ax2.set_xticks([0, 8, 16, 32, 48, 62])
plot_with_band(ax2, Ks_c_l, BA_c_l, SD_c_l, "Orthogonal Softmax", COL_LICS, "o")
plot_with_band(ax2, Ks_c_g, BA_c_g, SD_c_g, "Vanilla Softmax",        COL_GS,   "s")
plot_with_band(ax2, Ks_c_r, BA_c_r, SD_c_r, "Regularized Softmax",    COL_REG,  "d")
add_baseline(ax2, BASE_CROSS_BA_MEAN, BASE_CROSS_BA_STD)
ax2.set_xlabel("Number of selected channels (K)")
ax2.set_ylabel("Balanced Accuracy (Cross-)")
ax2.grid(True, lw=0.4, alpha=0.5)

# (c) Unique ratio vs K (Single)
ax3.set_xlim(0, 62); ax3.set_ylim(0.43, 1.02)
ax3.set_xticks([0, 8, 16, 32, 48, 62])
plot_unique(ax3, Ks_s_l, UR_s_l, "Orthogonal Softmax", COL_LICS, "o")
plot_unique(ax3, Ks_s_g, UR_s_g, "Vanilla Softmax",        COL_GS,   "s")
plot_unique(ax3, Ks_s_r, UR_s_r, "Regularized Softmax",    COL_REG,  "d")
ax3.set_xlabel("Number of selected channels (K)")
ax3.set_ylabel("Unique ratio (Within-)")
ax3.grid(True, lw=0.4, alpha=0.5)

# (d) Unique ratio vs K (Cross)
ax4.set_xlim(0, 62); ax4.set_ylim(0.43, 1.02)
ax4.set_xticks([0, 8, 16, 32, 48, 62])
plot_unique(ax4, Ks_c_l, UR_c_l, "Orthogonal Softmax", COL_LICS, "o")
plot_unique(ax4, Ks_c_g, UR_c_g, "Vanilla Softmax",        COL_GS,   "s")
plot_unique(ax4, Ks_c_r, UR_c_r, "Regularized Softmax",    COL_REG,  "d")
ax4.set_xlabel("Number of selected channels (K)")
ax4.set_ylabel("Unique ratio (Cross-)")
ax4.grid(True, lw=0.4, alpha=0.5)

# 在子图底部中央添加标签
ax1.text(0.5, -0.15, '(a)', fontsize=9, ha='center', va='top', transform=ax1.transAxes)
ax2.text(0.5, -0.15, '(b)', fontsize=9, ha='center', va='top', transform=ax2.transAxes)
ax3.text(0.5, -0.25, '(c)', fontsize=9, ha='center', va='top', transform=ax3.transAxes)
ax4.text(0.5, -0.25, '(d)', fontsize=9, ha='center', va='top', transform=ax4.transAxes)

# 顶部合并图例
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))

fig.tight_layout(rect=[0, 0, 1, 0.98], w_pad=1.2, h_pad=1.0)
plt.savefig("Fig_orthogonality_THU.pdf", bbox_inches="tight")
plt.show()