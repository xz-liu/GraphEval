from trainer import train_classifier, train_prompt_encoder
from args import args
from utils import *

if __name__ == '__main__':
    model_name = args.model_name
    model_type = args.model_type
    test_case_name = args.test_case_name
    which = args.which

    kg = get_kg(args)
    use_peft= not args.no_peft

    if use_peft:
        # check if the prompt encoder is already trained
        from os.path import exists
        from args.config import trained_prompt_encoder_path
        prompt_encoder_path = trained_prompt_encoder_path(model_name, test_case_name, which=which)

        if exists(prompt_encoder_path):
            print(f"Prompt encoder already trained for model {model_name} and test case {test_case_name}.")
            print(f"Path: {prompt_encoder_path}")
        else:
            train_prompt_encoder(kg, model_name=model_name, test_case_name=test_case_name, model_type=model_type)
    # train_prompt_encoder(kg, model_name=model_name, test_case_name=test_case_name, model_type=model_type)
    train_classifier(kg, model_name=model_name, test_case_name=test_case_name, model_type=model_type, use_peft=use_peft)