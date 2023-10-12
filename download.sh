if [ "$1" = "metaworld" ]; then
    wget -c https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/metaworld/model-24.pt
    mkdir -p results/mw
    mv model-24.pt results/mw/model-24.pt
    echo "Downloaded metaworld model"
elif [ "$1" = "ithor" ]; then
    wget -c https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/ithor/model-30.pt
    mkdir -p results/thor
    mv model-30.pt results/thor/model-30.pt
    echo "Downloaded ithor model"
elif [ "$1" = "bridge" ]; then
    wget -c https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/bridge/model-42.pt
    mkdir -p results/bridge
    mv model-42.pt results/bridge/model-42.pt
    echo "Downloaded bridge model"
else 
    echo "Options: {metaworld, ithor, bridge}"
fi