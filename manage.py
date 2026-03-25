#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# Add external TensorFlow path
if os.path.exists('C:\\tf_temp'):
    sys.path.insert(0, 'C:\\tf_temp')


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Liver_Image_Segmentation_Using_DeepLearning.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
