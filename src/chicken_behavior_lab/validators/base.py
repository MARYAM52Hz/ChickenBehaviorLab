"""
ChickenBehaviorLab Validation Base Classes
==========================================

Base interfaces for validating framework data models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")


class BaseValidator(ABC, Generic[T]):
    """
    Base validator for ChickenBehaviorLab data models.
    """

    @abstractmethod
    def validate(self, data: T) -> bool:
        """
        Validate the provided data.

        Returns
        -------
        bool
            True if the data is valid.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_or_raise(self, data: T) -> None:
        """
        Validate data and raise an exception if invalid.
        """
        raise NotImplementedError
