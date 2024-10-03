### for mmnist
- epochs 2000
- lr 1e-3

   ```sh
   python main.py --data mmnist --in_shape "(10,1,64,64)" --lr 5e-4 --scheduler onecycle --epochs 2000 --test_batch_size 16 --save_path InvVP_mmnist
   ```

### for taxibj
- epochs 50
- lr 1e-3

   ```sh
   python main.py --save_path InvVP_taxibj
   ```