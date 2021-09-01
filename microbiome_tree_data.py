from torch.utils.data import Dataset
import re
import numpy as np

class MicrobiomeTreeData(Dataset):
    def __init__(self, df, x_name, y_name, y_mapping, transform=None, dtype='float32'):
        self.df = df
        self.y_mapping = y_mapping

        # Extract the x and y we want to use
        self.x_df = self.df.filter(regex=r'^{:}[0-9]+$'.format(x_name), axis=1)
        if y_name in self.df.columns:
            self.y_df = self.df.loc[:, [y_name]]
        else:
            self.y_df = None
            print('y_name cannot be found in columns of the data, so y will be None')

        # Drop samples containing NaN
        if y_name in self.df.columns:
            mask_null = (self.x_df.isnull().any(axis=1) | self.y_df.isnull().any(axis=1))
            self.x_df = self.x_df.loc[~mask_null, :].reset_index(drop=True)
            self.y_df = self.y_df.loc[~mask_null, :].reset_index(drop=True)
        else:
            mask_null = self.x_df.isnull().any(axis=1)
            self.x_df = self.x_df.loc[~mask_null, :].reset_index(drop=True)
        if mask_null.sum() > 0:
            print('{:d} samples containing NaN are dropped'.format(mask_null.sum()))

        # Print label distribution in y
        if (y_name in self.df.columns) and (self.y_mapping is not None):
            print('Label distribution is as follows:')
            print(self.y_df[y_name].value_counts())

        # Encode labels in y to numbers
        if (y_name in self.df.columns) and (self.y_mapping is not None):
            assert set(self.y_df[y_name]) == set(self.y_mapping.keys())
            self.y_df[y_name] = self.y_df[y_name].replace(self.y_mapping)

        # Convert the data frames to numpy arrays of a particular data type
        self.x_data = self.x_df.values.astype(dtype)
        if x_name == 'rel':
            self.x_data = np.log(self.x_data + 1)

        self.y_data = self.y_df.values.astype(dtype) if y_name in self.df.columns else None

        # Create a mapping from leaf name to column index of x_data
        self.leaf_name_to_index = {re.match(r'^{:}(?P<leaf_name>[0-9]+)$'.format(x_name), col_name).group('leaf_name'): i
                                   for i, col_name in enumerate(self.x_df.columns)}

        self.transform = transform

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, i):
        x = self.x_data[i, :]
        if self.transform is not None:
            x = self.transform(x)

        if self.y_data is not None:
            y = self.y_data[i, :]
            return x, y
        else:
            return x
