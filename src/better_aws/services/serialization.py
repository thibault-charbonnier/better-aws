from typing import Any, Optional, Tuple
import pickle
from io import BytesIO


def _serialize_object(self,
                      obj: Any,
                      extension: Optional[str]) -> Tuple[bytes, str]:
        """
        Serialize a Python object to bytes, using the specified extension to determine the format.
        
        Supported extensions:
        - .pkl, .pickle: Pickle format (default)
        - .joblib, .jl: Joblib format
        - .skops: Skops format (preferable for sklearn models)

        Parameters:
        -----------
        obj: Any
            The Python object to serialize.
        extension: Optional[str]
            The file extension to determine the serialization format.
            If None, defaults to .pkl.

        Returns:
        --------
        Tuple[bytes, str]
            A tuple containing the serialized data as bytes and the corresponding content type.
        """
        ext = extension if extension else ".pkl"
    
        if ext in {".pkl", ".pickle"}:
            data = pickle.dumps(obj, protocol=self.pickle_protocol)
            data, ct = data, "application/octet-stream"
        elif ext in {".joblib", ".jl"}:
            try:
                import joblib
            except ImportError:
                raise ImportError(f"joblib is not installed but required for extension {ext}. To enable object serialization for `.joblib` and `.skops` please pip install `better-aws[objects]`.")
            bio = BytesIO()
            joblib.dump(obj, bio, compress=self.joblib_compress)
            data, ct = bio.getvalue(), "application/octet-stream"
        elif ext == ".skops":
            try:
                import skops.io as sio
            except ImportError:
                raise ImportError(f"skops is not installed but required for extension {ext}. To enable object serialization for `.joblib` and `.skops` please pip install `better-aws[objects]`.")
            data = sio.dumps(obj)
            data, ct = data, "application/octet-stream"
        else:
            raise ValueError(f"Unsupported object extension for python object: {ext}")

        return data, ct

def _deserialize_object(self,
                        data: bytes,
                        extension: Optional[str]) -> Any:
        """
        Deserialize bytes back into a Python object, using the specified extension to determine the format.
        
        Supported extensions:
        - .pkl, .pickle: Pickle format (default)
        - .joblib, .jl: Joblib format
        - .skops: Skops format (preferable for sklearn models)

        Parameters:
        -----------
        data: bytes
            The serialized data as bytes.
        extension: Optional[str]
            The file extension to determine the deserialization format.
            If None, defaults to .pkl.

        Returns:
        --------
        Any
            The deserialized Python object.
        """
        ext = extension if extension else ".pkl"

        if ext in {".pkl", ".pickle"}:
            obj = pickle.loads(data)
        elif ext in {".joblib", ".jl"}:
            try:
                import joblib
            except ImportError:
                raise ImportError(f"joblib is not installed but required for extension {ext}. To enable object serialization for `.joblib` and `.skops` please pip install `better-aws[objects]`.")
            bio = BytesIO(data)
            obj = joblib.load(bio)
        elif ext == ".skops":
            try:
                import skops.io as sio
            except ImportError:
                raise ImportError(f"skops is not installed but required for extension {ext}. To enable object serialization for `.joblib` and `.skops` please pip install `better-aws[objects]`.")
            obj = sio.loads(data)
        else:
            raise ValueError(f"Unsupported object extension for python object: {ext}")
        
        return obj