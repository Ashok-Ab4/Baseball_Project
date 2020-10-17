# let's create the transformer which contains the logic for calculating rolling batting average
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class RollingAverage(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(RollingAverage, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()
        dataset = dataset.withColumn(
            output_col, dataset[input_cols[0]] / dataset[input_cols[1]]
        )
        return dataset
