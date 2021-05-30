from tool.darknet2pytorch import Darknet
import struct
import sys

def convert_layer_name(name):
    parts = name.split(".")
    parts[0] = "model"
    parts[2] = ''.join([i for i in parts[2] if not i.isdigit()])
    return '.'.join(parts)

config = sys.argv[1]
model = Darknet(config)
weights = sys.argv[2]
model.load_weights(weights)

out_weights = sys.argv[3]
f = open(out_weights, 'w')

# Write number of entries
f.write(f"{len(model.state_dict().keys())}\n")
# Print all entries
print([convert_layer_name(module) for module in model.state_dict().keys()])
# Write to file
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write(f'{convert_layer_name(k)} {len(vr)} ')
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')