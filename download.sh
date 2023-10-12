

if $1 == "metaworld":
    wget https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/metaworld/model-24.pt -o results/mw/model-24.pt
else if $1 == "ithor":
    wget https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/ithor/model-30.pt -o results/thor/model-30.pt
else if $1 == "bridge":
    wget https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/bridge/model-42.pt -o results/bridge/model-42.pt
else 
    echo "Options: {metaworld, ithor, bridge}"
fi