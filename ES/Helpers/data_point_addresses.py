# Link to knowledge_base.ipynb:
# For detailed explanations, see:
# ../Knowledge_Base/knowledge_base.ipynb
# Specify the OPC UA server URL
server_url = "opc.tcp://000.000.000.000:4840"  # Replace with the actual server URL

# Specify the NodeIDs you want to read
node_ids = [
    # Replace with your NodeIDs; this is an example NodeID
    "ns=4;s=MAIN.VarPOU_LoTuS.LoTuS.DAK.BvL_Tanks.BvL_Tank_RZ.Tankheizung.BvL_T_T.sensorState.fValue",
]