featurecloud app build . fc-rand-forest && featurecloud test start --controller-host=http://localhost:8002 --query-interval=1 --app-image=fc-rand-forest --generic-dir tests/random_forest/iris/generic --channel "local" --client-dirs tests/random_forest/iris/3_clients_random_label/client1,tests/random_forest/iris/3_clients_random_label/client2,tests/random_forest/iris/3_clients_random_label/client3
