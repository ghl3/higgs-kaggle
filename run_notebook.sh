
# Run an ipython notebook via docker

docker build . -t higgs
docker run -t -i -p 8888:8888 -v $(PWD):/home/ubuntu higgs:latest sh -c 'jupyter notebook --ip 0.0.0.0 --no-browser --allow-root'
