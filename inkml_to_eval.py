import sys

if __name__ == '__main__':
    if len(sys.argv)  < 5:
        print("Usage: [Python] inkml_to_eval [inkml filelist] [path to output lg] [path to gt lg] [output name]")
    else:
        output = sys.argv[2]
        output = output if output[-1] == '/' else output + '/'
        gt = sys.argv[3]
        gt = gt if gt[-1] == '/' else gt + '/'
        with open(sys.argv[1], 'r') as read:
            with open(sys.argv[4], 'w') as out:
                for line in read:
                    temp = line.strip().replace('inkml', 'lg')
                    index = temp.find('/')
                    norel = temp[:index] + '_lg_norel' + temp[index:]
                    out.write(output + temp + ' ' + gt + norel + '\n')
