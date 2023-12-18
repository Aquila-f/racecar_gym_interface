#!/bin/bash

# Check if exactly two parameters are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [folder name]"
    exit 1
fi



OPENV="conda activate p310" # TODO: change to your own conda env name
FNAME="$1"


IMGPATH="record/${FNAME}_img/"
REALIMGPATH=$(realpath $IMGPATH)
POSSERVER="python server/pos_server.py --impath $REALIMGPATH"
PYGETPOS="python pos_check.py $FNAME"
TENSORB="tensorboard --logdir exp/$FNAME"
SESSION="$FNAME"

# 啟動 tmux 會話
tmux new-session -d -s $SESSION

# 在會話中創建一個新窗口
# tmux new-window -n my_window -t $SESSION:1

# 在新窗口中水平分割
tmux split-window -h -t $SESSION:0

# 選擇新分割的窗口，並在其中垂直分割
# tmux select-pane -t 0
# tmux split-window -v -t $SESSION:0.0

# 在第一個窗格中發送指令 echo 1
tmux send-keys -t $SESSION:0.0 'echo 1' C-m
tmux send-keys -t $SESSION:0.0 "$OPENV" C-m
tmux send-keys -t $SESSION:0.0 "$PYGETPOS" C-m

# 在第二個窗格中發送指令 echo 2
# tmux send-keys -t $SESSION:0.1 'echo 2' C-m
# tmux send-keys -t $SESSION:0.1 "$OPENV" C-m
# tmux send-keys -t $SESSION:0.1 "$TENSORB" C-m


# 在第二個窗格中發送指令 echo 2
tmux send-keys -t $SESSION:0.1 'echo 2' C-m
tmux send-keys -t $SESSION:0.1 'sleep 10 ' C-m
tmux send-keys -t $SESSION:0.1 "$OPENV" C-m
tmux send-keys -t $SESSION:0.1 "$POSSERVER" C-m


# 附加到創建的 tmux 會話
tmux attach-session -t $SESSION
