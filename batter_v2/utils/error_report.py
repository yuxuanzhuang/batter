import click
import os
import smtplib
from smtplib import SMTPException

MESSAGE_TEMPLATE = """From: batter <nobody@stanford.edu>
To: {recipient_name} <{recipient_email}>
Subject: {subject}

{message_body}
"""

MESSAGE_BODY_TEMPLATE = (
    "Hi there!\n\n"
    "I launched your job, {job_name}, {num_times} times, but it was unable to "
    "advance past stage {stage} and step {step}.\n\n"
    "Once you get it fixed, I'll automatically start taking care of it again.\n\n"
    "Thanks!\nmdstep\n\n"
    "P.S. Here's the folder: {folder}"
)

def sendmail(subject, body):
    sender = 'nobody@stanford.edu'  # Changed from mdstep@mdstep.com
    try:
        user = os.environ['USER']
        receivers = [f"{user}@stanford.edu"]
        message = MESSAGE_TEMPLATE.format(
            recipient_name=user,
            recipient_email=f"{user}@stanford.edu",
            subject=subject,
            message_body=body
        )
        with smtplib.SMTP('localhost') as smtp:
            smtp.sendmail(sender, receivers, message)
        print(f"Email sent successfully to {receivers}")
    except KeyError:
        print("Error: 'USER' environment variable not found.")
    except SMTPException as e:
        print(f"SMTP error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while sending mail: {e}")

@click.command(no_args_is_help='--help')
@click.option('--job_name', required=True, help="Name of the job.")
@click.option('--num_times', required=True, type=int, help="Number of times the job was attempted.")
@click.option('--stage', required=True, help="Stage where the job failed.")
@click.option('--step', required=True, help="Step where the job failed.")
@click.option('--folder', required=True, help="Folder containing job details.")
def error_report(job_name, num_times, stage, step, folder):
    subject = f"Error running job: {job_name}"
    body = MESSAGE_BODY_TEMPLATE.format(
        job_name=job_name,
        num_times=num_times,
        stage=stage,
        step=step,
        folder=folder
    )
    sendmail(subject, body)

if __name__ == '__main__':
    error_report()