

if $1 == "metaworld"; then
    wget https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/metaworld/model-24.pt -o results/mw/model-24.pt
elif $1 == "ithor"; then
    wget https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/ithor/model-30.pt -o results/thor/model-30.pt
elif $1 == "bridge"; then
    wget https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/bridge/model-42.pt -o results/bridge/model-42.pt
else 
    echo "Options: {metaworld, ithor, bridge}"
fi