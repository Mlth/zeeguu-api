import collections
import os

from matplotlib import pyplot as plt
import numpy as np
from zeeguu.recommender.utils import import_tf

tf = import_tf()

embeddings_path = "./zeeguu/recommender/embeddings/"
user_embeddings_path = f"{embeddings_path}user_embedding.npy"
article_embeddings_path = f"{embeddings_path}article_embedding.npy"

def train(embeddings, loss, metrics, num_iterations, learning_rate, plot_results, optimizer):
    """Trains the model.
    Args:
      iterations: number of iterations to run.
      learning_rate: optimizer learning rate.
      plot_results: whether to plot the results at the end of training.
      optimizer: the optimizer to use. Default to GradientDescentOptimizer.
    Returns:
      The metrics dictionary evaluated at the last iteration.
    """

    with loss.graph.as_default():
      opt = optimizer(learning_rate)
      train_op = opt.minimize(loss)
      local_init_op = tf.group(
          tf.variables_initializer(opt.variables()),
          tf.local_variables_initializer())
    session = tf.Session()
    with session.as_default():
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        tf.train.start_queue_runners()

    with session.as_default():
      local_init_op.run()
      iterations = []
      metrics = metrics or ({},)
      metrics_vals = [collections.defaultdict(list) for _ in metrics]

      # Train and append results.
      for i in range(num_iterations + 1):
        _, results = session.run((train_op, metrics))
        if (i % (num_iterations/10) == 0) or i == num_iterations:
          print("\r iteration %d: " % i + ", ".join(
                ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                end='')
          print()
          iterations.append(i)
          for metric_val, result in zip(metrics_vals, results):
            for k, v in result.items():
              metric_val[k].append(v)

      for k, v in embeddings.items():
        embeddings[k] = v.eval()

      if plot_results:
        # Plot the metrics.
        num_subplots = len(metrics)+1
        fig = plt.figure()
        fig.set_size_inches(num_subplots*10, 8)
        for i, metric_vals in enumerate(metrics_vals):
          ax = fig.add_subplot(1, num_subplots, i+1)
          for k, v in metric_vals.items():
            ax.plot(iterations, v, label=k)
          ax.set_xlim([1, num_iterations])
          ax.legend()

    save_embeddings(embeddings)

    return embeddings
    
def save_embeddings(embeddings):
    user_em = embeddings["user_id"]
    article_em = embeddings["article_id"]

    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)
        print(f"Folder '{embeddings_path}' created successfully.")

    with open(embeddings_path + "user_embedding.npy", 'wb' ) as f:
        np.save(f, user_em)

    with open(embeddings_path + "article_embedding.npy", 'wb' ) as f:
        np.save(f, article_em)