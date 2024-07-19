import numpy as np
from typing import Sequence, Any, Tuple


class Quaternion:
    """A class representing a quaternion (w, x, y, z)"""

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1:
            self._from_array(args[0])
        elif len(args) == 4:
            self._from_components(*args)
        elif len(args) == 0:
            self._from_components(1, 0, 0, 0)

    def _from_array(self, array: np.ndarray) -> None:
        try:
            array = np.array(array, dtype=float)
        except ValueError:
            raise ValueError(
                "Invalid type for quaternion array. Expecting sequence of floats (4)"
            )
        if not array.shape == (4,):
            raise ValueError(
                f"Invalid shape of quaternion array: {array.shape}. Expecting shape (4,)"
            )
        self._q = array

    def _from_components(self, w: float, x: float, y: float, z: float) -> None:
        self._q = np.array([w, x, y, z])

    def norm(self) -> float:
        return float(np.linalg.norm(self._q))

    def normalize(self) -> "Quaternion":
        return Quaternion(self._q / self.norm())

    def conjugate(self) -> "Quaternion":
        return Quaternion(self._q * np.array([1, -1, -1, -1]))

    def inverse(self) -> "Quaternion":
        return self.conjugate() / self.norm()

    def dot(self, other: "Quaternion") -> float:
        return np.dot(self._q, other._q)

    def angle(self, other: "Quaternion") -> float:
        return np.arccos(self.dot(other) / (self.norm() * other.norm()))

    def rotate(self, vector: Sequence) -> np.ndarray:
        if not len(vector) == 3:
            raise ValueError(
                f"Invalid shape of vector: {len(vector)}. Expecting shape (3,)"
            )
        return self * Quaternion(0, *vector) * self.inverse()

    def to_angle_axis(self) -> Tuple[float, np.ndarray]:
        angle = 2 * np.arccos(self.w)
        axis = self._q[:3] / np.sin(angle / 2)
        return angle, axis

    def to_euler(self) -> np.ndarray:
        w, x, y, z = self._q
        t0 = 2 * (w * x + y * z)
        t1 = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = 2 * (w * y - z * x)
        t2 = 1 if t2 > 1 else t2
        t2 = -1 if t2 < -1 else t2
        pitch = np.arcsin(t2)
        t3 = 2 * (w * z + x * y)
        t4 = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw])

    def to_matrix(self) -> np.ndarray:
        w, x, y, z = self._q
        return np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )

    @property
    def x(self) -> float:
        return self._q[1]

    @property
    def y(self) -> float:
        return self._q[2]

    @property
    def z(self) -> float:
        return self._q[3]

    @property
    def w(self) -> float:
        return self._q[0]

    @staticmethod
    def Identity() -> "Quaternion":
        return Quaternion(1, 0, 0, 0)

    # Operator overloading

    def __add__(self, other: "Quaternion") -> "Quaternion":
        if not isinstance(other, Quaternion):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'Quaternion' and '{type(other)}'"
            )
        return Quaternion(self._q + other._q)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        if not isinstance(other, Quaternion):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'Quaternion' and '{type(other)}'"
            )
        return Quaternion(self._q - other._q)

    def __mul__(self, other: "Quaternion" | Sequence | np.ndarray) -> Any:
        if isinstance(other, Quaternion):
            return self._multiply_quaternion(other)
        elif isinstance(other, (Sequence, np.ndarray)):
            return self._multiply_vector(other)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: 'Quaternion' and '{type(other)}'"
            )

    def __truediv__(self, other) -> "Quaternion":
        if isinstance(other, Quaternion):
            return self._multiply_quaternion(other.inverse())
        elif isinstance(other, (int, float)):
            return Quaternion(self._q / other)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for /: 'Quaternion' and '{type(other)}'"
            )

    def _multiply_quaternion(self, other: "Quaternion") -> "Quaternion":
        return Quaternion.quaternion_multiplication(self, other)

    def _multiply_vector(self, vector: Sequence | np.ndarray) -> np.ndarray:
        if not len(vector) == 3:
            raise ValueError(
                f"Invalid shape of vector: {len(vector)}. Expecting shape (3,)"
            )
        w, x, y, z = self._q
        vq = Quaternion(0, *vector)
        result = self * vq * self.inverse()
        return result._q[1:]

    def __iter__(self):
        return iter(self._q)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> float:
        if index < 0 or index > 3:
            raise IndexError(f"[Quaternion->Getter] Index out of range: {index}")
        return self._q[index]

    def __setitem__(self, index: int, value: float) -> None:
        if index < 0 or index > 3:
            raise IndexError(f"[Quaternion->Setter] Index out of range: {index}")
        self._q[index] = value

    def __str__(self) -> str:
        return f"Quaternion({self._q})"

    @staticmethod
    def rotation(vector_a: Sequence, vector_b: Sequence) -> "Quaternion":
        try:
            vector_a = np.array(vector_a, dtype=float)
            vector_b = np.array(vector_b, dtype=float)
        except ValueError:
            raise ValueError(
                "Invalid type for input vectors. Expecting sequence of floats (3)"
            )

        if not vector_a.shape == (3,) or not vector_b.shape == (3,):
            raise ValueError(
                f"Invalid shape of input vectors: {vector_a.shape}, {vector_b.shape}. Expecting shape (3,)"
            )

        vector_a = vector_a / np.linalg.norm(vector_a)
        vector_b = vector_b / np.linalg.norm(vector_b)
        axis = np.cross(vector_a, vector_b)
        angle = np.arccos(np.dot(vector_a, vector_b))
        return Quaternion.from_angle_axis(angle, axis)

    @staticmethod
    def from_angle_axis(angle: float, axis: Sequence) -> "Quaternion":
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        w = np.cos(angle / 2)
        xyz = np.sin(angle / 2) * axis
        return Quaternion(w, *xyz)

    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> "Quaternion":
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return Quaternion(w, x, y, z)

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> "Quaternion":
        if not matrix.shape == (3, 3):
            raise ValueError(
                f"Invalid shape of rotation matrix: {matrix.shape}. Expecting shape (3, 3)"
            )
        m00, m01, m02 = matrix[0]
        m10, m11, m12 = matrix[1]
        m20, m21, m22 = matrix[2]
        tr = m00 + m11 + m22
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (m21 - m12) / S
            y = (m02 - m20) / S
            z = (m10 - m01) / S
        elif m00 > m11 and m00 > m22:
            S = np.sqrt(1.0 + m00 - m11 - m22) * 2
            w = (m21 - m12) / S
            x = 0.25 * S
            y = (m01 + m10) / S
            z = (m02 + m20) / S
        elif m11 > m22:
            S = np.sqrt(1.0 + m11 - m00 - m22) * 2
            w = (m02 - m20) / S
            x = (m01 + m10) / S
            y = 0.25 * S
            z = (m12 + m21) / S
        else:
            S = np.sqrt(1.0 + m22 - m00 - m11) * 2
            w = (m10 - m01) / S
            x = (m02 + m20) / S
            y = (m12 + m21) / S
            z = 0.25 * S
        return Quaternion(w, x, y, z)

    @staticmethod
    def quaternion_multiplication(q1, q2: "Quaternion") -> "Quaternion":
        w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
        x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
        y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
        z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        return Quaternion(w, x, y, z)
